# python -m lilab.mmdet_dev.s3_segpkl_dilate2videoseg_canvas /mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/clips/2022-04-25_16-18-25_bwt_wwt_00time.mp4
import argparse
import os.path as osp
import pickle
import numpy as np
import pycocotools.mask as maskUtils
import cv2
import torch
from tqdm import tqdm
import glob
import lilab.cvutils.map_multiprocess_cuda as mmap_cuda
import itertools
import ffmpegcv
import copy

from lilab.cameras_setup import get_view_xywh_wrapper as get_view_xywh
video_path = [f for f in glob.glob('/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/TPH2HETxWT/*.mp4')
                if f[-5] not in '0123456789']
num_gpus = torch.cuda.device_count()


def get_hflips(nviews):
    if nviews==6:
        return [False for _ in range(6)]
    elif nviews==10:
        from lilab.cameras_setup import get_view_hflip
        return [False for _ in range(6)]
        return get_view_hflip()
    else:
        raise NotImplementedError


# class MyWorker(mmap_cuda.Worker):
class MyWorker():
    def compute(self, args):
        video_in = args
        self.cuda = getattr(self, 'cuda', 0)
        self.id = getattr(self, 'id', 0)
        vid = ffmpegcv.VideoReaderNV(video_in,
                                     gpu = self.cuda)
        segpkl  = osp.splitext(video_in)[0] + '.segpkl'
        segpkl_data = pickle.load(open(segpkl, 'rb'))
        segdata = segpkl_data['dilate_segdata']
        views = get_view_xywh(len(segdata))
        hflips = get_hflips(len(segdata))
        video_outb = video_in.replace('.mp4', f'_ratblack.mp4')
        video_outw = video_in.replace('.mp4', f'_ratwhite.mp4')
        # vidout =   [ffmpegcv.VideoWriterNV(video_outb, codec='h264', fps=vid.fps),
        #             ffmpegcv.VideoWriterNV(video_outw, codec='h264', fps=vid.fps)]
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        vidout =   [cv2.VideoWriter(video_outb, fourcc, 30.0, (3840,3200)),
                    cv2.VideoWriter(video_outw, fourcc, 30.0, (3840,3200))]
        class_names = ['rat_black', 'rat_white']
        nclass = len(class_names)
        mask_container =   [[None, None, None],
                            [None, None, None],
                            [None, None, None],
                            [None, np.zeros((800,1280),dtype=np.uint8), np.zeros((800,1280),dtype=np.uint8)]]
        mask_index = {0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2),
                      6:(2,0), 7:(2,1), 8:(2,2), 9:(3,0)}
        print('hello')
        for iframe, frame in enumerate(tqdm(vid, position=int(self.id), 
                                                desc='worker[{}]'.format(self.id))):
            # frame_cuda = torch.from_numpy(frame.copy()).cuda().half()
            # maskiclass_canvas = [copy.deepcopy(mask_container) for _ in range(nclass)]
            # for iview in range(len(views)):
            #     segdata_iview = segdata[iview][iframe]
            #     crop_xywh = views[iview]
            #     x, y, w, h = crop_xywh
            #     for iclass in range(nclass):
            #         mask = segdata_iview[1][iclass]
            #         mask = maskUtils.decode(mask)[:,:,0]
            #         if mask.shape[0] != h or mask.shape[1] != w:
            #             mask = cv2.resize(mask, (h,w), interpolation=cv2.INTER_NEAREST)
            #         # # cuda
            #         # maskcuda = torch.from_numpy(mask).cuda().half()
            #         index = mask_index[iview]
            #         maskiclass_canvas[iclass][index[0]][index[1]] = mask
            
            ## draw canvas
            # maskiclass_canvas_a = np.array(maskiclass_canvas)
            # maskiclass_canvas_b = torch.from_numpy(maskiclass_canvas_a).cuda(self.cuda).half()
            # maskiclass_canvas_n = maskiclass_canvas_b.permute(0,1,3,2,4)
            # shape_n = maskiclass_canvas_n.shape
            # maskiclass_canvas_b = maskiclass_canvas_n.reshape(shape_n[0],
            #                                             shape_n[1]*shape_n[2],
            #                                             shape_n[3]*shape_n[4])
            
            # frame_cuda_w_mask = frame_cuda[None,...] * maskiclass_canvas_b[...,None]

            # for crop_xywh, hflip in zip(views, hflips):
            #     if not hflip: continue
            #     for iclass in range(nclass):
            #         x, y, w, h = crop_xywh
            #         im1 = frame_cuda_w_mask[iclass][y:y+h, x:x+w]
            #         frame_cuda_w_mask[iclass][y:y+h, x:x+w] = torch.flip(im1, dims=[1])

            # for iclass in range(nclass):
            #     vidout[iclass].write(frame_cuda_w_mask[iclass].type(torch.uint8).cpu().numpy())

            ## draw canvas
            break
        
        for _ in tqdm(range(1000)):
            for iclass in range(nclass):
                vidout[iclass].write(frame)
            pass

        vid.release()
        for iclass in range(nclass):
            vidout[iclass].release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, default=None, help='path to video or folder')
    args = parser.parse_args()

    video_path = args.video_path
    assert osp.exists(video_path), 'video_path not exists'
    if osp.isfile(video_path):
        video_path = [video_path]
    elif osp.isdir(video_path):
        video_path = glob.glob(osp.join(video_path, '*.mp4'))
        assert len(video_path) > 0, 'no video found'
    else:
        raise ValueError('video_path is not a file or folder')

    args_iterable = video_path

    # init the workers pool
    # mmap_cuda.workerpool_init(range(num_gpus), MyWorker)
    # mmap_cuda.workerpool_compute_map(args_iterable)
    worker = MyWorker()
    for i in range(len(args_iterable)):
        worker.compute(args_iterable[i])
