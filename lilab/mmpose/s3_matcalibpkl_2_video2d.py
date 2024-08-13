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

def plot_video(video, crop_xywh, pts2d_black, pts2d_white, iview, postfix):
    gpu, s = get_gpu()
    # vid = ffmpegcv.VideoCaptureNV(video, crop_xywh=crop_xywh, gpu=gpu)
    vid = ffmpegcv.noblock(ffmpegcv.VideoCaptureNV, video, crop_xywh=crop_xywh,gpu=gpu)
    assert len(pts2d_black) == len(pts2d_white), 'len(pts2d_black) != len(pts2d_white)'
    maxlen = len(pts2d_black)
    if len(vid) == maxlen:
        pass
    elif len(vid) > maxlen:
        print('Warning: the annotation is shorter than the video!')
    elif len(vid) < maxlen:
        raise ValueError('len(vid) < maxlen')
    postfix = f'_{postfix}' if postfix else ''
    output_file = osp.splitext(video)[0] + f'_{iview}_sktdraw{postfix}.mp4'
    # vidout = ffmpegcv.VideoWriterNV(output_file, codec='h264', fps=vid.fps, gpu=gpu)
    #vidout = ffmpegcv.noblock(ffmpegcv.VideoWriterNV, output_file, codec='h264', fps=vid.fps, gpu=gpu)
    vidout = ffmpegcv.noblock(ffmpegcv.VideoWriterNV, output_file, codec='h264', fps=vid.fps,gpu=gpu, resize=(800,600))
    for i, frame, pts2d_b_now, pts2d_w_now in zip(tqdm.tqdm(range(maxlen)), vid, pts2d_black, pts2d_white):
        if not np.all(np.isnan(pts2d_b_now)):
            frame = cv_plot_skeleton_aframe(frame, pts2d_b_now, name = 'black')
        if not np.all(np.isnan(pts2d_w_now)):
            frame = cv_plot_skeleton_aframe(frame, pts2d_w_now, name = 'white')
        frame = cv2.putText(frame, str(i), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        vidout.write(frame)

    vid.release()
    vidout.release()
    s.release()


def main(kptpkl, iview, postfix, maxlen=None, setupname='carl'):
    pkldata = pickle.load(open(kptpkl, 'rb'))
    video = osp.dirname(osp.abspath((kptpkl))) + '/' + osp.basename(kptpkl).split('.')[0] + '.mp4'
    views = get_view_xywh_wrapper(setupname)
    crop_xywh = views[iview]
    kpt_rats_xy = pkldata['keypoints_xy_ba'][iview]
    pts2d_black = kpt_rats_xy[:, 0, :, :]
    pts2d_white = kpt_rats_xy[:, 1, :, :] if kpt_rats_xy.shape[1]>=2 else pts2d_black*np.nan
    if maxlen is not None and maxlen < len(pts2d_black):
        pts2d_black = pts2d_black[:maxlen]
        pts2d_white = pts2d_white[:maxlen]

    plot_video(video, crop_xywh, pts2d_black, pts2d_white, iview, postfix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the skeleton of a video')
    parser.add_argument('kptpkl', type=str, help='kptpkl file')
    parser.add_argument('--iview', type=int, default=iview, help='view index')
    parser.add_argument('--postfix', type=str, default='', help='postfix of the output video')
    parser.add_argument('--maxlen', type=int, default=None, help='maxlen of the video')
    parser.add_argument('--setupname', type=str, default='carl', help='setupname of the video')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    args = parser.parse_args()
    main(args.kptpkl, args.iview, args.postfix, args.maxlen, args.setupname)
