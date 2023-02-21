# python -m lilab.mmdet_dev.s4_segpkl_com3d_to_video A.mp4
import argparse
import os.path as osp
import pickle
import numpy as np
import cv2
import torch
import copy
from tqdm import tqdm
import glob
import lilab.cvutils.map_multiprocess_cuda as mmap_cuda
import itertools
import os
import ffmpegcv
from lilab.multiview_scripts_dev.s6_calibpkl_predict import CalibPredict
from lilab.mmlab_scripts.show_pkl_seg_video_fast import get_mask_colors
video_path = [f for f in glob.glob('/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/TPH2HETxWT/*.mp4')
                if f[-5] not in '0123456789' and 'mask' not in f]

from lilab.mmlab_scripts.show_pkl_seg_video_fast import default_mask_colors

mask_colors = torch.Tensor(get_mask_colors())

vox_size = 180
preview_resize = (1280, 800)
verts = np.array([[1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, 1],
                    [-1, 1, 1],
                    [-1, -1, 1],
                    [1, -1, 1]])
bones = np.array([[0, 1], [1, 2], [2, 3], [3,0],
                  [0, 4], [1, 5], [2, 6], [3,7],
                  [4, 5], [5, 6], [6, 7], [7,4]])

def draw_point_color(frame, keypoints_xy, color, radius=3):
    keypoint_i_xy = keypoints_xy.reshape(-1, 2)
    keypoint_i_xy = keypoint_i_xy[~np.isnan(keypoint_i_xy[:,0])]
    # draw keypoints
    for xy in keypoint_i_xy:
        cv2.circle(frame, tuple(xy.astype(np.int32)), radius, color.tolist(), -1)

def draw_cube_color(frame, cube_iframe_xy, color):
    assert cube_iframe_xy.ndim==3 and cube_iframe_xy.shape[1]==8
    # draw cube
    for iview in range(cube_iframe_xy.shape[0]):
        cube_iframe_iview_xy = cube_iframe_xy[iview]
        cube_int = cube_iframe_iview_xy.astype(np.int32)
        for idx in bones:
            if np.any(np.isnan(cube_iframe_iview_xy[idx])): continue
            cv2.line(frame, tuple(cube_int[idx[0]]), tuple(cube_int[idx[1]]), color.tolist(), 2)

def p2d_to_canvas(p2d, views_xywh, scale_wh):
    assert len(p2d)==len(views_xywh)
    for iview in range(len(p2d)):
        p2d[iview] += np.array(views_xywh[iview][:2])
    p2d[..., 0] *= scale_wh[0]
    p2d[..., 1] *= scale_wh[1]
    return p2d


class MyWorker(mmap_cuda.Worker):
# class MyWorker():
    def compute(self, args):
        video_in, vox_size, maxlen = args
        self.cuda = getattr(self, 'cuda', 0)
        self.id = getattr(self, 'id', 0)
        pkl_path = video_in.replace('.mp4', '.segpkl')
        mask_video = video_in.replace('.mp4', '_mask.mp4')
        if osp.exists(mask_video): 
            vid = ffmpegcv.VideoCaptureNV(mask_video, gpu=self.cuda, pix_fmt='rgb24')
        else:
            vid = ffmpegcv.VideoCaptureNV(video_in, gpu=self.cuda, pix_fmt='rgb24', resize=preview_resize)
        pkl_data = pickle.load(open(pkl_path, 'rb'))
        assert {'coms_3d', 'coms_2d'} < set(pkl_data.keys())
        views_xywh = np.array(pkl_data["views_xywh"])
        coms_2d = pkl_data["coms_2d"]
        coms_3d = pkl_data["coms_3d"]
        if not maxlen:
            maxlen = len(coms_3d)
            print(maxlen)
        coms_2d = coms_2d[:maxlen]
        coms_3d = coms_3d[:maxlen]
        origin_width = np.max(views_xywh[:,0]+views_xywh[:,2])
        origin_height = np.max(views_xywh[:,1]+views_xywh[:,3])
        scale_wh = vid.width/origin_width, vid.height/origin_height

        calibPredict = CalibPredict(pkl_data)
        coms_2d = p2d_to_canvas(coms_2d, views_xywh, scale_wh) # (nview, nsample, nclass, 2)
        nview, nsample, nclass, _ = coms_2d.shape
        cube_xyz = coms_3d[None, ...] + verts[:,None,None,:]*vox_size/2  # (nsample, nclass, 3) + (8, 3) = (8, nsample, nclass, 3)
        cube_xyz[:,:,:,2] = np.clip(cube_xyz[:,:,:,2], 0, None) # set z to 0 if z<0
        cube_xy = calibPredict.p3d_to_p2d(cube_xyz)    # (nview, 8, nsample, nclass, 2)
        vox_size=int(vox_size)
        cube_xy = p2d_to_canvas(cube_xy, views_xywh, scale_wh) # (nview, 8, nsample, nclass, 2)
        video_out = video_in.replace('.mp4', f'_com3d_vol{vox_size}.mp4')
        vidout = ffmpegcv.VideoWriterNV(video_out, 
                                        gpu = self.cuda,
                                        codec='h264', 
                                        fps=vid.fps, 
                                        pix_fmt='rgb24')
        
        for i in tqdm(range(min(len(vid), maxlen)), position=int(self.id), 
                                        desc='worker[{}]'.format(self.id)):
            ret, frame = vid.read()
            for iclass in range(nclass):
                # draw coms center
                keypoints_xy = coms_2d[:, i, iclass, :]  # (nview, 2)
                color = default_mask_colors[iclass][0]//1.5
                draw_point_color(frame, keypoints_xy, color, 5)

                # draw coms cube
                cube_xy_i = cube_xy[:,:,i,iclass,:] # (nview, 8, 2)
                draw_cube_color(frame, cube_xy_i, color)
            frame = cv2.putText(frame, str(i), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
            vidout.write(frame)

        vid.release()
        vidout.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, default=None, help='path to video or folder')
    parser.add_argument('--vox_size', type=float, default=vox_size, help='voxel total box size')
    parser.add_argument('--maxlen', type=int, default=None, help='maxlen of the video')
    args = parser.parse_args()

    video_path = args.video_path
    assert osp.exists(video_path), 'video_path not exists'
    assert args.vox_size>0, 'voxel size should >0'
    if osp.isfile(video_path):
        video_path = [video_path]
    elif osp.isdir(video_path):
        video_path = glob.glob(osp.join(video_path, '*.mp4'))
        assert len(video_path) > 0, 'no video found'
    else:
        raise ValueError('video_path is not a file or folder')

    args_iterable = [(v, args.vox_size, args.maxlen) for v in video_path]
    num_gpus = min([torch.cuda.device_count()*4, len(args_iterable)])
    # init the workers pool
    mmap_cuda.workerpool_init(range(num_gpus), MyWorker)
    mmap_cuda.workerpool_compute_map(args_iterable)

    # worker = MyWorker()
    # for args in args_iterable:
    #     worker.compute(args)
    