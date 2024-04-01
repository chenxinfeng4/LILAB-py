# python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_faster  /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/SOLO/
# %%
import argparse
import numpy as np
import torch
import itertools
import json
import matplotlib.pyplot as plt

from lilab.mmpose_dev.a2_convert_mmpose2engine import findcheckpoint_trt
from lilab.cameras_setup import get_view_xywh_wrapper
from lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_realtimecam import (
    transform_preds
)
from lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_faster import  MyWorker
import lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_faster as S1BV2MFF
import os.path as osp
import glob
import lilab.cvutils.map_multiprocess_cuda as mmap_cuda
from lilab.multiview_scripts_dev.comm_functions import non_max_suppression


num_gpus = min([torch.cuda.device_count(), 4])

config_dict = { 1:'/home/liying_lab/chenxinfeng/DATA/mmpose/res50_coco_ball_640x480_coupleball.py'}


def get_max_preds_triple(heatmaps:torch.Tensor):
    # sum the C=3 dimension and keep the dimension (N, 1, H, W)
    heatmaps = heatmaps.sum(dim=1, keepdim=True)
    heatmaps = heatmaps.detach().cpu().numpy() # (N, C=1, H=320, W=512)
    N, K, _, W = heatmaps.shape # K==1
    peaks = np.zeros((N, K, 2, 3), dtype=np.float32)
    for i in range(N):
        for k in range(K):
            heatmap = heatmaps[i, k]
            plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
            
            
            peaks[i, k] = non_max_suppression(heatmap, num_peaks=2, w=10)
    peaks_squeeze = peaks[:,0]
    preds = peaks_squeeze[..., :2]
    maxvals = peaks_squeeze[..., [2]]
    return preds, maxvals


def post_cpu(camsize, heatmap, center, scale, views_xywh, img_preview, calibobj):
    N, K, H, W = heatmap.shape
    preds, maxvals = get_max_preds_triple(heatmap) # (N, K, 2), (N, K, 1)
    preds = transform_preds(
                preds, center, scale, [W, H], use_udp=False)
    maxvals[:] = np.min(maxvals, axis=1, keepdims=True)  #coupleball 共享最小 P 值
    keypoints_xyp = np.concatenate((preds, maxvals), axis=-1) #(N, K, xyp)
    #keypoints_xyp = np.array([sort_peaks(k) for k in keypoints_xyp]) #coupleball 排序


    return keypoints_xyp

S1BV2MFF.post_cpu = post_cpu


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, help='path to video or folder')
    parser.add_argument('--pannels', default=1, help='crop views')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    arg = parser.parse_args()

    views_xywh = get_view_xywh_wrapper(arg.pannels)
    nviews = len(views_xywh)
    video_path, config, checkpoint = arg.video_path, arg.config, arg.checkpoint
    if config is None:
        config = config_dict[nviews]
    if checkpoint is None:
        checkpoint = findcheckpoint_trt(config, 'latest.full_fp16.engine')
    print("config:", config)
    print("checkpoint:", checkpoint)

    assert osp.exists(video_path), 'video_path not exists'
    if osp.isfile(video_path):
        video_path = [video_path]
    elif osp.isdir(video_path):
        video_path = glob.glob(osp.join(video_path, '*.mp4'))
        video_path = [v for v in video_path
                      if 'sktdraw' not in v and
                         'com3d' not in v and
                         'mask' not in v]
        assert len(video_path) > 0, 'no video found'
    else:
        raise ValueError('video_path is not a file or folder')
    
    args_iterable = itertools.product([config], video_path, [checkpoint], [views_xywh])
    # for args in args_iterable:
    #     print(args)
    # exit(0)

    # init the workers pool
    # mmap_cuda.workerpool_init(range(num_gpus), MyWorker)
    # mmap_cuda.workerpool_compute_map(args_iterable)

    worker = MyWorker()
    for args in args_iterable:
        worker.compute(args)
