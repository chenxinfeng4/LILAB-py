# python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full  A.mp4
# %%
import argparse
import os
import re
import os.path as osp
import glob
import mmcv
import numpy as np
import tqdm
import torch
import pickle
from mmpose.apis import init_pose_model
from mmpose.datasets import DatasetInfo
# import lilab.cvutils.map_multiprocess_cuda as mmap_cuda
import ffmpegcv
from ffmpegcv.video_info import get_info
from torch2trt import TRTModule
import itertools
from lilab.mmpose_dev.a2_convert_mmpose2engine import findcheckpoint_trt
from torch2trt.torch2trt import torch_dtype_from_trt
from lilab.cameras_setup import get_view_xywh_wrapper
from lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_realtimecam import (
    get_max_preds_gpu, transform_preds, pre_cpu, mid_gpu, box2cs, 
    preview_resize, DataSet as OldDataSet
)

config_dict = {6:'/home/liying_lab/chenxinfeng/DATA/mmpose/hrnet_w32_coco_ball_512x512.py',
          10:'/home/liying_lab/chenxinfeng/DATA/mmpose/res50_coco_ball_512x512.py',
          9:'/home/liying_lab/chenxinfeng/DATA/mmpose/res50_coco_ball_512x320_cam9.py',
          4: '/home/liying_lab/chenxinfeng/DATA/mmpose/res50_coco_ball_512x512_ZJF.py'}

class DataSet(OldDataSet): 
    def __init__(self, vid, cfg, views_xywh, c_channel_in=1):
        super().__init__(vid, cfg, views_xywh, c_channel_in)
        self.vid = vid
        self.views_xywh = views_xywh
        self.c_channel_in = c_channel_in
        
    def __iter__(self):
        vread = {1:self.vid.read_gray, 3:self.vid.read}[self.c_channel_in]
        while True:
            ret, img = vread()
            if not ret: raise StopIteration
            img_preview = np.zeros((preview_resize[1],preview_resize[0],3), dtype=np.uint8)
            # img_NCHW = img.ravel()[self.coord_NCHW_idx_ravel]
            img_NHWC = []
            for (x,y,w,h) in self.views_xywh:
                img_NHWC.append(img[y:y+h,x:x+w])
            img_NHWC = np.stack(img_NHWC, axis=0)
            img_NCHW = img_NHWC.transpose(0,3,1,2)
            if img_NCHW.shape[1]==1: img_NCHW = np.repeat(img_NCHW, 3, axis=1)
            yield img_NCHW, img_preview


class MyWorker():
    def __init__(self, config, video_file, checkpoint, views_xywh):
        super().__init__()
        self.id = getattr(self, 'id', 0)
        self.cuda = getattr(self, 'cuda', 0)
        pose_model = init_pose_model(
                        config, checkpoint=None, device='cpu')
        self.pose_model = pose_model
        cfg = pose_model.cfg
        feature_in_wh = np.array(cfg.data_cfg['image_size'])
        self.checkpoint = checkpoint
        views_xywh_np = np.array(views_xywh)
        views_xyxy_np = np.concatenate((views_xywh_np[:,:2], views_xywh_np[:,:2]+views_xywh_np[:,2:]), axis=1)
        canvas_xyxy = np.concatenate((views_xyxy_np[:,:2].min(axis=0), views_xyxy_np[:,2:].max(axis=0)))
        canvas_xywh = np.concatenate((canvas_xyxy[:2], canvas_xyxy[2:]-canvas_xyxy[:2]))
        assert canvas_xywh[:2].tolist()==[0,0], 'Views_xywh should have a view start at (x=0, y=0)'
        view_w, view_h = views_xywh_np[0,[2,3]]
        assert (views_xywh_np[:,2:]==(view_w, view_h)).all(), 'Views_xywh should have same view size'
        scale_shrink_w = view_w / feature_in_wh[0]
        scale_shrink_h = view_h / feature_in_wh[1]
        canvas_xywh_shrink = np.round(canvas_xywh / np.array([scale_shrink_w, scale_shrink_h, scale_shrink_w, scale_shrink_h])).astype(int)
        views_xywh_shrink = np.round(views_xywh_np / np.array([scale_shrink_w, scale_shrink_h, scale_shrink_w, scale_shrink_h])).astype(int)
        vid = ffmpegcv.VideoCaptureNV(video_file, pix_fmt='rgb24', crop_xywh=canvas_xywh.tolist(), resize=canvas_xywh_shrink[2:].tolist())
        c_channel_in=1 if vid.pix_fmt=='nv12' else 3
        dataset = DataSet(vid, cfg, views_xywh_shrink, c_channel_in=c_channel_in)
        dataset.center, dataset.scale = box2cs(np.array([0,0,view_w,view_h]), feature_in_wh)
        self.video_file = video_file
        self.views_xywh = views_xywh
        self.dataset = dataset
        self.dataset_iter = iter(self.dataset)
        print("Well setup VideoCapture")

    def compute(self, args):
        views_xywh = self.views_xywh
        dataset, dataset_iter = self.dataset, self.dataset_iter
        center, scale = dataset.center, dataset.scale
        count_range = range(dataset.__len__()) if hasattr(dataset, '__len__') else itertools.count()

        with torch.cuda.device(self.cuda):
            trt_model = TRTModule()
            trt_model.load_from_engine(self.checkpoint)
            idx = trt_model.engine.get_binding_index(trt_model.input_names[0])
            input_dtype = torch_dtype_from_trt(trt_model.engine.get_binding_dtype(idx))
            input_shape = tuple(trt_model.context.get_binding_shape(idx))
            if input_shape[0]==-1:
                assert input_shape[1:]==dataset.coord_NCHW_idx_ravel.shape[1:]
                input_shape = dataset.coord_NCHW_idx_ravel.shape
            else:
                assert input_shape==dataset.coord_NCHW_idx_ravel.shape
            img_NCHW = np.ones(input_shape)
            img_preview = np.zeros((*preview_resize,3))
            heatmap = mid_gpu(trt_model, img_NCHW, input_dtype)
            # img_NCHW, img_preview = pre_cpu(dataset_iter)
            keypoints_xyp = []
            for idx, _ in enumerate(tqdm.tqdm(count_range, desc='loading'), start=-1):
                heatmap_wait = mid_gpu(trt_model, img_NCHW, input_dtype)
                img_NCHW_next, img_preview_next = pre_cpu(dataset_iter)
                torch.cuda.current_stream().synchronize()
                img_NCHW, img_preview = img_NCHW_next, img_preview_next
                heatmap = heatmap_wait
                if idx<=-1: continue
                kpt2d = post_cpu(None, heatmap, center, scale, views_xywh, img_preview, None)
                keypoints_xyp.append(kpt2d)

            heatmap = mid_gpu(trt_model, img_NCHW, input_dtype)
            kpt2d = post_cpu(None, heatmap, center, scale, views_xywh, img_preview, None)
            keypoints_xyp.append(kpt2d)
        
        # save data to pickle file
        keypoints_xyp = np.array(keypoints_xyp).transpose(1,0,2,3)#(nview, T, 1, 3)
        # assert np.mean(keypoints_xyp[...,2].ravel()<0.4) < 0.1, 'Too many nan in keypoints_xyp !'
        assert np.median(keypoints_xyp[...,2])>0.4, 'Too many nan in keypoints_xyp !'

        info = {'vfile': self.video_file, 'nview': len(self.views_xywh), 'fps':  self.dataset.vid.fps}
        outpkl = os.path.splitext(self.video_file)[0] + '.matpkl'
        outdict = dict(
            keypoints = keypoints_xyp,
            views_xywh = self.views_xywh,
            info = info
        )
        pickle.dump(outdict, open(outpkl, 'wb'))
        print('python -m lilab.multiview_scripts_dev.s2_matpkl2ballpkl',
            outpkl, '--time 1 2 3 4 5')
        print('python -m lilab.multiview_scripts_dev.s5_show_calibpkl2video', outpkl)


def post_cpu(camsize, heatmap, center, scale, views_xywh, img_preview, calibobj):
    N, K, H, W = heatmap.shape
    preds, maxvals = get_max_preds_gpu(heatmap)
    preds = transform_preds(
                preds, center, scale, [W, H], use_udp=False)
    keypoints_xyp = np.concatenate((preds, maxvals), axis=-1) #(N, xyp)
    return keypoints_xyp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, help='path to video or folder')
    parser.add_argument('--pannels', type=int, default=4, help='crop views')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    arg = parser.parse_args()

    views_xywh = get_view_xywh_wrapper(arg.pannels)
    video_path, config, checkpoint = arg.video_path, arg.config, arg.checkpoint
    if config is None:
        config = config_dict[arg.pannels]
    if checkpoint is None:
        checkpoint = findcheckpoint_trt(config, 'latest.full.engine')
    print("config:", config)
    print("checkpoint:", checkpoint)

    worker = MyWorker(config, video_path, checkpoint,  views_xywh)
    worker.compute(None)

