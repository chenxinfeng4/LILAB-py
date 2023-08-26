#!/usr/bin/python
# python -m lilab.cvutils.crop_videofast A.mp4

import os
import argparse
import cv2
from lilab.cameras_setup import get_view_xywh_wrapper
crop_tbg = None
crop_tdur = '00:10:00'
codec = 'h264_nvenc'
encoder = 'h264_cuvid'
nviews = 'els'


def xywh2whxy(xywh):
    whxy = (xywh[2], xywh[3], xywh[0], xywh[1])
    return whxy

def fun_crop_mp4(filename, whxy_list):
    ffmpeg_args = ''
    if crop_tbg:
        ffmpeg_args = ffmpeg_args + f' -ss {crop_tbg}'
    if crop_tdur:
        ffmpeg_args = ffmpeg_args + f' -t {crop_tdur}'
        
    # convert videos
    cropstr_list = [f'crop={whxy[0]}:{whxy[1]}:{whxy[2]}:{whxy[3]}' for whxy in whxy_list]
    filterstr = ';'.join([f'[0:v]{cropstr}[v{i}]' for i,cropstr in enumerate(cropstr_list)])
    
    # check & downsample frame rate <=10fps
    cap= cv2.VideoCapture(filename)
    framespersecond= cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fps30 = min([30, framespersecond])

    # convert        
    ffout_list = [f'-map "[v{i}]" -y -r {fps30} -c:v {codec} {ffmpeg_args} -preset p6 -tune hq -b:v 1M "{filename[:-4]}_output_{i+1}.mp4"'
            for i in range(len(whxy_list))]

    ffoutstr = ' '.join(ffout_list)

    mystr = f'ffmpeg -i "{filename}" -filter_complex "{filterstr}" {ffoutstr}'
    print(mystr)
    out = os.system(mystr)


def main(vfile):
    views_xywh = get_view_xywh_wrapper(nviews)
    whxy_list = [xywh2whxy(xywh) for xywh in views_xywh]
    fun_crop_mp4(vfile, whxy_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vfile', type=str)
    args = parser.parse_args()
    main(args.vfile)