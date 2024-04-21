# python -m lilab.cvutils_new.extract_frames_pannel_crop_random 2024-01-31_15-38-50_JF1_3_and_JF4_2_soft_oak.mp4  --npick 30 --setupname carl
import argparse
import os 
import cv2
import numpy as np
import tqdm
import os.path as osp
import sys
import glob
import ffmpegcv
from lilab.cameras_setup import get_view_xywh_wrapper
import random

numframe_to_extract = 100
# numframe_to_extract = 100
maxlength = 3000 #前3000zhen
# maxlength = 1000
frame_dir = "outframes"
frame_min_interval = 1  #间隔多少zhen



def extract_random(video_input, setupname, numframe_to_extract, maxlength):
    views = get_view_xywh_wrapper(setupname)
    nviews = len(views)
    dirname,filename=os.path.split(video_input)
    nakefilename = os.path.splitext(filename)[0]
    cap = ffmpegcv.VideoCapture(video_input)
    os.makedirs(os.path.join(dirname, frame_dir), exist_ok = True)
    length = cap.count
    length = min([maxlength, length-1]) if maxlength else length-1
    downsample_length = length // frame_min_interval
    # np.random.seed(0)
    idxframe_to_extract = set(np.random.permutation(downsample_length)[:numframe_to_extract]*frame_min_interval + 5)
    idxframe_max = max(idxframe_to_extract)
    # for i in tqdm.tqdm(range(10000)):
    #     ret, frame = cap.read()

    for iframe in tqdm.tqdm(range(length)):
        ret, frame = cap.read()
        if not ret: break
        if iframe>idxframe_max: break
        if iframe not in idxframe_to_extract: continue

        ipannel = random.randint(0, nviews-1)
        crop_xywh = views[ipannel]
        x, y, w, h = crop_xywh
        im1 = frame[y : y + h, x : x + w]
        filename = os.path.join(dirname, frame_dir, nakefilename + f'_pannel{ipannel}_{iframe:06}.jpg')
        cv2.imwrite(filename, im1, [int(cv2.IMWRITE_JPEG_QUALITY),100])
        
    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract image')
    parser.add_argument('video_path', type=str, default=None, help='path to image or folder')
    parser.add_argument('--setupname', type=str, default='ana', help='camera setup name')
    parser.add_argument('--npick', type=int, default=numframe_to_extract)
    args = parser.parse_args()

    video_path = args.video_path
    assert osp.exists(video_path), 'video_path not exists'
    if osp.isfile(video_path):
        video_path = [video_path]
    elif osp.isdir(video_path):
        video_path = [f for f in glob.glob(osp.join(video_path, '*.mp4'))
                        if f[-4] not in '0123456789']
        assert len(video_path) > 0, 'no video found'
    else:
        raise ValueError('video_path is not a file or folder')

    # read config_video.py
    for video_input in video_path:
        extract_random(video_input, args.setupname, args.npick, maxlength)
    
    print("Succeed")
