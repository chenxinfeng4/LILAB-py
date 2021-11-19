#!/usr/bin/python
'''
Author: your name
Date: 2021-09-28 14:14:27
LastEditTime: 2021-10-13 19:04:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \BCNete:\cxf\crop_video.py
'''
# !pyinstaller -F crop_image.py -i mypython.ico
# chenxinfeng
# ------使用方法------
# 直接拖动文件夹到EXE中
import sys
import os
from glob import glob 
import cxfguilib as cg
from PIL import Image

'''
vim config_video.py
## CROP_VIDEO.EXE
# crop_xywh = [640*1,480*1, 640,480]
w,h = 800, 600
crop_xywh = [[w*0,h*0,w,h],
             [w*1,h*0,w,h],
             [w*2,h*0,w,h],
             [w*0,h*1,w,h],
             [w*1,h*1,w,h],
             [w*2,h*1,w,h]]

keepXeqY = False
## EXTRACT_FRAME.EXE
numframe_to_extract = 200
maxlength = 10000
'''
#xywh = [50,0,600,585]
#folder = r"E:\cxf\Videos\20210601 LS RAT"

def xywh2whxy(xywh, keepXeqY=True):
    if keepXeqY:
        maxXY = max(xywh[2:])
        xywh[2] = xywh[3] = maxXY
    whxy = (xywh[2], xywh[3], xywh[0], xywh[1])
    return whxy

def convert_folder_to_mp4(folder, whxy, postfix=None):
    os.chdir(folder)
    filenamesList = glob(r'*.jpg') +  glob(r'*.png')
    filenamesList = [f for f in filenamesList if 'output' not in f]
    
    # check filenameList is sorted. If not, rise an error!
    if len(filenamesList) == 0:
        print("Folder contain no JPG/JPEG/PNG images!")
        return

    # convert videos
    for filename in filenamesList:
        outfileformat = "{}_output_"+str(postfix)+".png" if postfix else "{}_output.png"
        outfilename = outfileformat.format(filename[:-4])
        im = Image.open(filename)
        left, top, right, bottom = whxy[2], whxy[3], whxy[0]+whxy[2], whxy[1]+whxy[3]
        im1 = im.crop((left, top, right, bottom))
        im1.save(outfilename)

if __name__ == '__main__':
    n = len(sys.argv)
    if n == 1:
        folder = cg.uigetfolder()
        if folder == None:
            exit()
        else:
            sys.argv.append(folder)
            
    print(sys.argv[1:])

    video_foldname = sys.argv[1]
    config = cg.getfoldconfigpy(video_foldname)
    assert hasattr(config, 'crop_xywh'), 'Need [crop_xywh] in [config_video.py]'
    keepXeqY = getattr(config, 'keepXeqY', True)

    if type(config.crop_xywh[0]) == list:
        for i, crop_xywh in enumerate(config.crop_xywh):
            whxy = xywh2whxy(crop_xywh, keepXeqY)
            convert_folder_to_mp4(video_foldname, whxy, i+1)
    else:
        whxy = xywh2whxy(config.crop_xywh, keepXeqY)
        convert_folder_to_mp4(video_foldname, whxy)
    print("Succeed")
 