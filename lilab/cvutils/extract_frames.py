#!/usr/bin/python
# !pyinstaller -F extract_frames.py -i mypython.ico
# chenxinfeng
# ------使用方法------
# 直接拖动文件夹/视频到EXE中

import os
import cv2
import numpy as np
import tqdm
import sys
import cxfguilib as cg
from glob import glob

numframe_to_extract = 200
maxlength = 10000
frame_dir = "outframes"

def extract(video_input, numframe_to_extract, maxlength):
    dirname,filename=os.path.split(video_input)
    nakefilename = os.path.splitext(filename)[0]
    cap = cv2.VideoCapture(video_input)
    os.makedirs(frame_dir, exist_ok = True)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length = min([maxlength, length-1])
    idxframe_to_extract = set(np.random.permutation(length)[:numframe_to_extract])
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
    n = len(sys.argv)
    if n == 1:
        folder = cg.uigetfolder()
        if folder == None:
            exit()
        else:
            sys.argv.append(folder)
            
    print(sys.argv[1:])
    
    # check input is file or folder
    file_or_folderpath = sys.argv[1]
    if os.path.isfile(file_or_folderpath):
        video_foldname, f =os.path.split(file_or_folderpath)
        os.chdir(video_foldname)
        filenamesList = [f]
    elif os.path.isdir(file_or_folderpath):
        video_foldname = file_or_folderpath
        os.chdir(video_foldname)
        filenamesList = glob(r'*.avi') + glob(r'*.mp4') + glob(r'*.mkv')
        assert len(filenamesList), "Folder contain no AVI/MP4/MKV videos!"
    else:
        assert False, 'Input should be FILE or FOLDER'
        
    # read config_video.py
    try:
        config = cg.getfoldconfigpy('.')
        numframe_to_extract = getattr(config, 'numframe_to_extract', 200)
        maxlength = getattr(config, 'maxlength', 10000)
    except:
        pass
    
    for video_input in filenamesList:
        extract(video_input, numframe_to_extract, maxlength)
    
    print("Succeed")
