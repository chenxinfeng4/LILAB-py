#!/usr/bin/python
# python -m lilab.cvutils.extract_frames xxx
# !pyinstaller -F extract_frames.py -i mypython.ico
# chenxinfeng
# ------使用方法------
# 直接拖动文件夹/视频到EXE中

import os
import cv2
import numpy as np
import tqdm
from glob import glob
import argparse
import ffmpegcv

numframe_to_extract = 100
maxlength = 10000
frame_dir = "outframes"
frame_min_interval = 20

def extract(video_input, numframe_to_extract=numframe_to_extract, maxlength=maxlength):
    dirname,filename=os.path.split(video_input)
    nakefilename = os.path.splitext(filename)[0]
    cap = ffmpegcv.VideoCaptureNV(video_input)
    os.makedirs(frame_dir, exist_ok = True)
    length = min([maxlength, cap.count-1])
    downsample_length = length // frame_min_interval
    idxframe_to_extract = set(np.random.permutation(downsample_length)[:numframe_to_extract]*frame_min_interval)
    idxframe_max = max(idxframe_to_extract)

    for iframe in tqdm.tqdm(range(length)):
        ret, frame = cap.read()
        if not ret: break
        if iframe>idxframe_max: break
        if iframe not in idxframe_to_extract: continue
        filename = os.path.join(dirname, frame_dir, nakefilename + '_{0:06}.jpg'.format(iframe))
        cv2.imwrite(filename, frame)
        
    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_or_folder', type=str, help='video input')
    args = parser.parse_args()
    
    # check input is file or folder
    file_or_folderpath = args.video_or_folder
    if os.path.isfile(file_or_folderpath):
        video_foldname, f =os.path.split(os.path.abspath(file_or_folderpath))
        os.chdir(video_foldname)
        filenamesList = [f]
    elif os.path.isdir(file_or_folderpath):
        video_foldname = file_or_folderpath
        os.chdir(video_foldname)
        filenamesList = glob(r'*.avi') + glob(r'*.mp4') + glob(r'*.mkv')
        assert len(filenamesList), "Folder contain no AVI/MP4/MKV videos!"
    else:
        assert False, 'Input should be FILE or FOLDER'
    
    for video_input in filenamesList:
        extract(video_input)
    
    print("Succeed")
