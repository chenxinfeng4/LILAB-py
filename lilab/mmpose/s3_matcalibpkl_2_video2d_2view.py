# python -m lilab.mmpose.s3_matcalibpkl_2_video2d A.kptpkl
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

gpu = 0
ngpu = 4

def plot_video(video, crop_xywh, pts2d_b1, pts2d_w1, pts2d_b2, pts2d_w2, postfix):
    gpu, s = get_gpu()
    ratio = 2
    # (1280, 1600) / 1.5 = (854, 1066)
    resize_w = int(1280 / ratio // 2 * 2)
    resize_h = int(800*2 / ratio // 2 * 2)
    pts2d_b1 /= ratio
    pts2d_w1 /= ratio
    pts2d_b2[...,1]+=800
    pts2d_w2[...,1]+=800
    pts2d_b2 /= ratio
    pts2d_w2 /= ratio

    vid = ffmpegcv.VideoCaptureNV(video, crop_xywh=crop_xywh, resize=(resize_w, resize_h), gpu=gpu)
    assert len(pts2d_b1) == len(pts2d_w1) == len(pts2d_b2) == len(pts2d_w2), 'len(pts2d_black) != len(pts2d_white)'
    maxlen = len(pts2d_b1)
    if len(vid) == maxlen:
        pass
    elif len(vid) > maxlen:
        print('Warning: the annotation is shorter than the video!')
    elif len(vid) < maxlen:
        raise ValueError('len(vid) < maxlen')
    postfix = f'_{postfix}' if postfix else ''
    output_file = osp.splitext(video)[0] + f'_sktdraw{postfix}.mp4'
    vidout = ffmpegcv.VideoWriterNV(output_file, codec='h264', fps=vid.fps, gpu=gpu)

    for i, frame in zip(tqdm.tqdm(range(maxlen)), vid):
        if not np.all(np.isnan(pts2d_b1[i])):
            frame = cv_plot_skeleton_aframe(frame, pts2d_b1[i], name = 'black')
        if not np.all(np.isnan(pts2d_w1[i])):
            frame = cv_plot_skeleton_aframe(frame, pts2d_w1[i], name = 'white')
        if not np.all(np.isnan(pts2d_b2[i])):
            frame = cv_plot_skeleton_aframe(frame, pts2d_b2[i], name = 'black')
        if not np.all(np.isnan(pts2d_w2[i])):
            frame = cv_plot_skeleton_aframe(frame, pts2d_w2[i], name = 'white')
        frame = cv2.putText(frame, str(i), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        vidout.write(frame)

    vid.release()
    vidout.release()
    s.release()



def main(kptpkl, postfix, maxlen=None):
    pkldata = pickle.load(open(kptpkl, 'rb'))
    video = osp.join(osp.dirname(kptpkl), osp.basename(kptpkl).split('.')[0]  + '.mp4')
    crop_xywh = [1280,0,1280, 800*2]
    assert len(pkldata['keypoints_xy_ba'])==9

    iview=1
    kpt_rats_xy = pkldata['keypoints_xy_ba'][iview]
    pts2d_black = kpt_rats_xy[:, 0, :, :]
    pts2d_white = kpt_rats_xy[:, 1, :, :] if kpt_rats_xy.shape[1]>=2 else pts2d_black*np.nan
    if maxlen is not None and maxlen < len(pts2d_black):
        pts2d_black = pts2d_black[:maxlen]
        pts2d_white = pts2d_white[:maxlen]

    pts2d_b1, pts2d_w1 = pts2d_black, pts2d_white

    iview=4
    kpt_rats_xy = pkldata['keypoints_xy_ba'][iview]
    pts2d_black = kpt_rats_xy[:, 0, :, :]
    pts2d_white = kpt_rats_xy[:, 1, :, :] if kpt_rats_xy.shape[1]>=2 else pts2d_black*np.nan
    if maxlen is not None and maxlen < len(pts2d_black):
        pts2d_black = pts2d_black[:maxlen]
        pts2d_white = pts2d_white[:maxlen]
    pts2d_b2, pts2d_w2 = pts2d_black, pts2d_white

    plot_video(video, crop_xywh, pts2d_b1, pts2d_w1, pts2d_b2, pts2d_w2, postfix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the skeleton of a video')
    parser.add_argument('kptpkl', type=str, help='kptpkl file')
    parser.add_argument('--postfix', type=str, default='', help='postfix of the output video')
    parser.add_argument('--maxlen', type=int, default=None, help='maxlen of the video')
    args = parser.parse_args()
    main(args.kptpkl, args.postfix, args.maxlen)
