# python -m lilab.cvutils_new.extract_frames_pannel_crop_fromjson /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/202402_scpp_behavior/conditioning_room2/errorframe/1/out_2024-01-31_17-07-24_JM3_3_and_JM1_3_paper_cotton.mp4/out.json
#/home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/cvutils_new/extract_frames_pannel_crop_fromjson.py
import argparse
import os
import cv2
import numpy as np
import tqdm
import os.path as osp
import json
import ffmpegcv
from lilab.cameras_setup import get_view_xywh_wrapper
import pandas as pd
import glob
import os.path as osp


pannels_xywh = get_view_xywh_wrapper('carl')
frame_dir = "outframes_raw_2"


def parser_json(json_file, dir_name=None):
    ipannels = osp.basename(osp.dirname(osp.dirname(json_file)))
    ipannels = int(ipannels) 
    assert ipannels in range(9)
    data = json.load(open(json_file, 'r'))
    vfiles = list(data.keys())
    if dir_name is None:
        dir_name = osp.dirname(osp.dirname(json_file))
    filenames_dict = {filename:osp.join(dirpath, filename) for dirpath, _, filenames in os.walk(dir_name) for filename in filenames}
    for vfile in vfiles:
        full_vfile = filenames_dict[vfile]
        # full_vfile = os.path.join(dir_name, vfile)
        assert osp.exists(full_vfile), 'video_path not exists'
        idxframes = [int(idxframe) for idxframe in data[vfile]]
        ready_to_extract(full_vfile, idxframes, dir_name, ipannels)


def ready_to_extract(video_input, idxframe_to_extract, dirname, ipannels):
    idxframe_max = max(idxframe_to_extract)
    _, filename=os.path.split(video_input)
    nakefilename = os.path.splitext(filename)[0]
    os.makedirs(os.path.join(dirname, frame_dir), exist_ok = True)
    cap = ffmpegcv.VideoCaptureNV(video_input, pix_fmt='nv12')
    length = idxframe_max+1
    for iframe in tqdm.tqdm(range(length)):
        ret, frame = cap.read()
        if not ret: break
        if iframe>idxframe_max: break
        if iframe not in idxframe_to_extract: continue
        if iframe not in idxframe_to_extract: continue
        frame_name = nakefilename + '_' + str(iframe) + '.jpg'
        x,y,w,h = pannels_xywh[ipannels]
        imgDataCrop = frame[y:y+h,x:x+w]
        filename = os.path.join(dirname, frame_dir, nakefilename + '_{:06}.jpg'.format(iframe))
        cv2.imwrite(filename, imgDataCrop, [int(cv2.IMWRITE_JPEG_QUALITY),95])
    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str)
    parser.add_argument('--dir_name', type=str, default=None)
    args = parser.parse_args()
    parser_json(args.json, args.dir_name)
    print('Done')
