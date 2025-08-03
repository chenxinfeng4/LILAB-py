# python -m lilab.mmpose.s3_matcalibpkl_2_video2d A.kptpkl --iview 0
import numpy as np
import os.path as osp
import pickle
import tqdm
import ffmpegcv
import argparse
import cv2
from lilab.multiview_scripts.rat2d_kptvideo import cv_plot_skeleton_aframe
from lilab.cameras_setup import get_view_xywh_wrapper
from lilab.paralleltool.gpuquery import get_gpu

pkl_file = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/clips/2022-04-25_15-44-04_bwt_wwt_00time_0.kptpkl'
setupname = 'carl'
iview = 0
ngpu = 4

def plot_video(video, crop_xywh, pts2d_rats_l, iview, postfix, gpu_id=None):

    if gpu_id is None:
        # gpu, s = get_gpu()
        gpu = np.random.randint(0, ngpu)
    else:
        gpu = gpu_id % ngpu
    # vid = ffmpegcv.VideoCaptureNV(video, crop_xywh=crop_xywh, gpu=gpu)
    vid = ffmpegcv.noblock(ffmpegcv.VideoCaptureNV, video, crop_xywh=crop_xywh,gpu=gpu)
    nrats = len(pts2d_rats_l)
    maxlen = len(pts2d_rats_l[0])
    assert all(len(pts2d_rat)==maxlen for pts2d_rat in pts2d_rats_l)
    
    if len(vid) == maxlen:
        pass
    elif len(vid) > maxlen:
        print('Warning: the annotation is shorter than the video!')
    elif len(vid) < maxlen:
        raise ValueError('len(vid) < maxlen')
    postfix = f'_{postfix}' if postfix else ''
    output_file = osp.splitext(video)[0] + f'_{iview}_sktdraw{postfix}.mp4'
    vidout = ffmpegcv.noblock(ffmpegcv.VideoWriterNV, output_file, codec='h264', fps=vid.fps,gpu=gpu) #, resize=(800,600)
    for i, frame in zip(tqdm.tqdm(range(maxlen), position=gpu_id), vid):
        pts2d_rats_now = [pts2d_rat[i] for pts2d_rat in pts2d_rats_l]
        for irat, pts2d_rat_now in enumerate(pts2d_rats_now):
            frame = cv_plot_skeleton_aframe(frame, pts2d_rat_now, name = irat)
        frame = cv2.putText(frame, str(i), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        vidout.write(frame)

    vid.release()
    vidout.release()


def main(kptpkl, iview, postfix, maxlen=None, setupname='carl', gpu_id=None):
    pkldata = pickle.load(open(kptpkl, 'rb'))
    video = osp.dirname(osp.abspath((kptpkl))) + '/' + osp.basename(kptpkl).split('.')[0] + '.mp4'
    views = get_view_xywh_wrapper(setupname)
    crop_xywh = views[iview]
    kpt_rats_xy = pkldata['keypoints_xy_ba'][iview] # shape: (nframes, nrats, 2) => (nrat, maxlen, 2)
    if maxlen is not None and maxlen < len(kpt_rats_xy):
        kpt_rats_xy = kpt_rats_xy[:maxlen]
    pts2d_rats_l = [kpt_rats_xy[:, i] for i in range(kpt_rats_xy.shape[1])]

    plot_video(video, crop_xywh, pts2d_rats_l, iview, postfix, gpu_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the skeleton of a video')
    parser.add_argument('kptpkl', type=str, help='kptpkl file')
    parser.add_argument('--iview', type=int, default=iview, help='view index')
    parser.add_argument('--postfix', type=str, default='', help='postfix of the output video')
    parser.add_argument('--maxlen', type=int, default=None, help='maxlen of the video')
    parser.add_argument('--setupname', type=str, default='carl', help='setupname of the video')
    parser.add_argument('--gpu-id', type=int, default=None, help='gpu index')
    args = parser.parse_args()
    main(args.kptpkl, args.iview, args.postfix, args.maxlen, args.setupname, args.gpu_id)
