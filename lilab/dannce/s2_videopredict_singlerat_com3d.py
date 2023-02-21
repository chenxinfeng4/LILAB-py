# python -m lilab.dannce.s2_videopredict_singlerat_com3d --pannel 4 --video_file xx.mp4 --ballcalib xxx.calibpkl --config xx.py 
# similar to lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_realtimecam
"""
cd /home/liying_lab/chenxinfeng/DATA/mmpose
python -m lilab.dannce.s2_videopredict_singlerat_com3d --pannel 4 \
    --video_file /mnt/liying.cibr.ac.cn_Data_Temp/ZJF_lab/0921/male.mp4 \
    --ballcalib /mnt/liying.cibr.ac.cn_Data_Temp/ZJF_lab/ball2.calibpkl \
    --config res50_coco_com2d_512x320_ZJF.py
"""
import argparse
import numpy as np
import tqdm
import torch
from torch2trt import TRTModule
import os
import pickle
from lilab.mmpose_dev.a2_convert_mmpose2engine import findcheckpoint_trt
from lilab.cameras_setup import get_view_xywh_wrapper
import itertools
from sklearn.impute import KNNImputer
from torch2trt.torch2trt import torch_dtype_from_trt
from scipy.ndimage import convolve1d

from lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_realtimecam import (
    get_max_preds_gpu, transform_preds, pre_cpu, mid_gpu,
    preview_resize, MyWorker as OldMyWorker
)

config_dict = {6:'/home/liying_lab/chenxinfeng/DATA/mmpose/hrnet_w32_coco_ball_512x512.py',
          10:'/home/liying_lab/chenxinfeng/DATA/mmpose/res50_coco_ball_512x512.py',
          4: r"E:\mmpose\res50_coco_ball_512x320_ZJF.py"}

class MyWorker(OldMyWorker):
    def compute(self, args):
        views_xywh = self.views_xywh
        dataset, dataset_iter = self.dataset, self.dataset_iter
        center, scale = dataset.center, dataset.scale
        count_range = range(dataset.__len__()) if hasattr(dataset, '__len__') else itertools.count()
        pbar = tqdm.tqdm(count_range, desc='loading')

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
            img_NCHW = np.zeros(input_shape)
            img_preview = np.zeros((*preview_resize,3))
            heatmap = mid_gpu(trt_model, img_NCHW, input_dtype)
            calibobj = self.calibobj
            keypoints_xyz_ba, keypoints_xyp = [], []
            for idx, _ in enumerate(count_range, start=-1):
                pbar.update(1)
                heatmap_wait = mid_gpu(trt_model, img_NCHW, input_dtype)
                img_NCHW_next, img_preview_next = pre_cpu(dataset_iter)
                torch.cuda.current_stream().synchronize()
                img_NCHW, img_preview = img_NCHW_next, img_preview_next
                heatmap = heatmap_wait
                if idx<=-1: continue
                # if idx>5000: break
                kpt3d, kpt2d = post_cpu(self.camsize, heatmap, center, scale, views_xywh, img_preview, calibobj)
                keypoints_xyz_ba.append(kpt3d)
                keypoints_xyp.append(kpt2d)
                

            heatmap = mid_gpu(trt_model, img_NCHW, input_dtype)
            kpt3d, kpt2d = post_cpu(self.camsize, heatmap, center, scale, views_xywh, img_preview, calibobj)
            keypoints_xyz_ba.append(kpt3d)
            keypoints_xyp.append(kpt2d)
        
        # save data to pickle file
        keypoints_xyz_ba = np.array(keypoints_xyz_ba)#(T,1,3)
        keypoints_xy_ba  = calibobj.p3d_to_p2d(keypoints_xyz_ba) #(nview, T, 1, 2)
        keypoints_xyp = np.array(keypoints_xyp).transpose(1,0,2,3)#(nview, T, 1, 3)
        if np.sum(np.isnan(keypoints_xyz_ba)) / keypoints_xyz_ba.size > 0.1:
            print('Warning too many nan in keypoints_xyz_ba !')
            return

        coms_3d = KNNImputer(n_neighbors=3).fit_transform(keypoints_xyz_ba.reshape(-1,3)).reshape(keypoints_xyz_ba.shape) #(T,1,3)
        # moving average the coms_3d by window_size=5 using scipy
        coms_3d = convolve1d(coms_3d, np.ones(5)/5, axis=0, mode='nearest')
        coms_2d = calibobj.p3d_to_p2d(coms_3d)  #(nview, T, 1, 2)
        
        info = {'vfile': self.video_file, 'nview': len(self.views_xywh), 'fps':  self.vid.fps}
        outpkl = os.path.splitext(self.video_file)[0] + '.segpkl'
        outdict = dict(
            coms_3d = coms_3d,
            coms_2d = coms_2d,
            ba_poses = calibobj.poses,
            views_xywh = self.views_xywh,
            info = info)
        pickle.dump(outdict, open(outpkl, 'wb'))

        outpkl = os.path.splitext(self.video_file)[0] + '_com3d.matcalibpkl'
        outdict = dict(
            keypoints_xyz_ba = keypoints_xyz_ba,
            keypoints_xyz_baglobal = keypoints_xyz_ba,
            keypoints_xy_ba = keypoints_xy_ba,
            keypoints = keypoints_xyp,
            views_xywh = self.views_xywh,
            ba_poses = calibobj.poses,
            info = info
        )
        pickle.dump(outdict, open(outpkl, 'wb'))
        print('python -m lilab.multiview_scripts_new.s5_show_calibpkl2video', outpkl)


def post_cpu(camsize, heatmap, center, scale, views_xywh, img_preview, calibobj):
    N, K, H, W = heatmap.shape
    preds, maxvals = get_max_preds_gpu(heatmap)
    preds = transform_preds(
                preds, center, scale, [W, H], use_udp=False)
    keypoints_xyp = np.concatenate((preds, maxvals), axis=-1) #(N, xyp)

    # thr
    thr = 0.4    
    indmiss = keypoints_xyp[...,2] < thr
    keypoints_xyp[indmiss] = np.nan
    keypoints_xy = keypoints_xyp[...,:2]

    # ba
    if calibobj is not None:
        keypoints_xyz_ba = calibobj.p2d_to_p3d(keypoints_xy)
        # keypoints_xy_ba = calibobj.p3d_to_p2d(keypoints_xyz_ba)
    else:
        keypoints_xyz_ba = np.ones((*keypoints_xy.shape[1:-1], 3)) * np.nan
        # keypoints_xy_ba = keypoints_xy * np.nan
    return keypoints_xyz_ba, keypoints_xyp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pannels', type=int, default=4, help='crop views')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--ballcalib', type=str, default=None)
    arg = parser.parse_args()

    views_xywh = get_view_xywh_wrapper(arg.pannels)
    config, checkpoint, ballcalib = arg.config, arg.checkpoint, arg.ballcalib
    if config is None:
        config = config_dict[arg.pannels]
    if checkpoint is None:
        checkpoint = findcheckpoint_trt(config, trtnake='latest.full.engine')
    assert arg.video_file, "--video_file should be set."
    assert ballcalib, "--ballcalib should be set"
    print("config:", config)
    print("checkpoint:", checkpoint)

    worker = MyWorker(config, arg.video_file, checkpoint, ballcalib, views_xywh)
    worker.compute(None)
