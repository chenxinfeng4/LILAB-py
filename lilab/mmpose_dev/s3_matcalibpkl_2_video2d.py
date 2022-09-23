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

pkl_file = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/clips/2022-04-25_15-44-04_bwt_wwt_00time_0.kptpkl'

iview = 0

def plot_video(video, crop_xywh, pts2d_black, pts2d_white, iview):
    vid = ffmpegcv.VideoCaptureNV(video, crop_xywh=crop_xywh)
    assert len(vid) == len(pts2d_black) == len(pts2d_white), 'vid and pts2d must have the same length'
    output_file = osp.splitext(video)[0] + f'_{iview}_sktdraw.mp4'
    vidout = ffmpegcv.VideoWriterNV(output_file, codec='h264', fps=vid.fps)

    for i, (frame, pts2d_b_now, pts2d_w_now) in enumerate(zip(tqdm.tqdm(vid), pts2d_black, pts2d_white)):
        if not np.all(np.isnan(pts2d_b_now)):
            frame = cv_plot_skeleton_aframe(frame, pts2d_b_now, name = 'black')
        if not np.all(np.isnan(pts2d_w_now)):
            frame = cv_plot_skeleton_aframe(frame, pts2d_w_now, name = 'white')
        frame = cv2.putText(frame, str(i), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        vidout.write(frame)

    vid.release()
    vidout.release()


def main(kptpkl, iview):
    pkldata = pickle.load(open(kptpkl, 'rb'))
    video = osp.dirname(osp.abspath((kptpkl))) + '/' + osp.basename(pkldata['info']['vfile'])
    views = get_view_xywh_wrapper(len(pkldata['keypoints_xy_ba']))
    crop_xywh = views[iview]
    kpt_rats_xy = pkldata['keypoints_xy_ba'][iview]
    pts2d_black = kpt_rats_xy[:, 0, :, :]
    pts2d_white = kpt_rats_xy[:, 1, :, :]

    plot_video(video, crop_xywh, pts2d_black, pts2d_white, iview)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the skeleton of a video')
    parser.add_argument('kptpkl', type=str, help='kptpkl file')
    parser.add_argument('--iview', type=int, default=iview, help='view index')
    args = parser.parse_args()
    main(args.kptpkl, args.iview)
