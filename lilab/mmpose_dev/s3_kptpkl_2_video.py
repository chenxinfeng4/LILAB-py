# python -m lilab.mmpose.s3_kptpkl_2_video x2022-04-25_16-18-25_bwt_wwt_02time_0.kptpkl
import numpy as np
import os.path as osp
import pickle
import tqdm
import ffmpegcv
from ffmpegcv.ffmpeg_reader_hflip import FFmpegReaderHFlip
import argparse
from lilab.multiview_scripts.rat2d_kptvideo import cv_plot_skeleton_aframe
from lilab.cameras_setup import get_view_xywh_1280x800x10 as get_view_xywh
from lilab.cameras_setup import get_view_hflip

pkl_file = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/clips/2022-04-25_16-18-25_bwt_wwt_00time_1.kptpkl'


def plot_video(video, iview, pts2d_black, pts2d_white):
    crop_xywh = get_view_xywh()[iview]
    hflip = get_view_hflip()[iview]
    vid = FFmpegReaderHFlip.VideoReader(video, crop_xywh=crop_xywh, hflip=hflip)
    assert len(vid) == len(pts2d_black) == len(pts2d_white), 'vid and pts2d must have the same length'
    output_file = osp.splitext(video)[0] + f'_{iview}_sktdraw.mp4'
    vidout = ffmpegcv.VideoWriterNV(output_file, codec='h264', fps=vid.fps)

    for i, (frame, pts2d_b_now, pts2d_w_now )in enumerate(zip(tqdm.tqdm(vid), pts2d_black, pts2d_white)):
        if i>200:return
        if not np.all(np.isnan(pts2d_b_now)):
            frame = cv_plot_skeleton_aframe(frame, pts2d_b_now, name = 'black')
        if not np.all(np.isnan(pts2d_w_now)):
            frame = cv_plot_skeleton_aframe(frame, pts2d_w_now, name = 'white')
        vidout.write(frame)

    vid.release()
    vidout.release()


def main(kptpkl):
    pkldata = pickle.load(open(kptpkl, 'rb'))
    video = osp.dirname(osp.abspath((kptpkl))) + '/' + osp.basename(pkldata['info']['vfile'])
    iview = int(kptpkl[-8])
    kpt_rats = pkldata[str(iview)]
    thr = 0.2
    kpt_rats_xy = kpt_rats[..., :2]
    kpt_rats_p  = kpt_rats[..., 2]
    kpt_rats_xy[kpt_rats_p<thr] = np.nan
    pts2d_black = kpt_rats_xy[:, 0, :, :]
    pts2d_white = kpt_rats_xy[:, 1, :, :]

    plot_video(video, iview, pts2d_black, pts2d_white)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the skeleton of a video')
    parser.add_argument('kptpkl', type=str, help='kptpkl file')
    args = parser.parse_args()
    main(args.kptpkl)
