#!/usr/bin/python
'''
Author: your name
Date: 2021-09-28 14:14:27
LastEditTime: 2021-10-13 19:04:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \BCNete:\cxf\crop_video.py
'''
# !pyinstaller -F crop_video.py -i mypython.ico
# chenxinfeng
# ------使用方法------
# 直接拖动文件夹到EXE中

'''
w, h = 800, 600
crop_xywh = [[w*0,h*0,w,h],
             [w*1,h*0,w,h],
             [w*2,h*0,w,h],
             [w*0,h*1,w,h],
             [w*1,h*1,w,h],
             [w*2,h*1,w,h]]
crop_tbg = ''
crop_tdur = '00:02:00'
'''
import sys
import os
from glob import glob 
import cv2
import platform

#xywh = [50,0,600,585]
#folder = r"E:\cxf\Videos\20210601 LS RAT"
crop_tbg = None
crop_tdur = None
def xywh2whxy(xywh, keepXeqY=True):
    if keepXeqY:
        maxXY = max(xywh[2:])
        xywh[2] = xywh[3] = maxXY
    whxy = (xywh[2], xywh[3], xywh[0], xywh[1])
    return whxy

def convert_folder_to_mp4(folder, whxy, postfix=None):
    os.chdir(folder)
    filenamesList = glob(r'*.avi') + glob(r'*.mp4') + glob(r'*.mkv')
    filenamesList = [f for f in filenamesList if 'output' not in f]
    
    # check filenameList is sorted. If not, rise an error!
    if len(filenamesList) == 0:
        print("Folder contain no AVI/MP4/MKV videos!")
        return
        
    # check FFMPEG exist
    ffmpeg_path = 'ffmpeg'

    ffmpeg_args = ''
    if crop_tbg:
        ffmpeg_args = ffmpeg_args + f' -ss {crop_tbg}'
    if crop_tdur:
        ffmpeg_args = ffmpeg_args + f' -t {crop_tdur}'
        
    # convert videos
    for filename in filenamesList:
        # check & downsample frame rate <=10fps
        cap= cv2.VideoCapture(filename)
        framespersecond= cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        fps10 = min([10, framespersecond])
    
        # convert
        if platform.system()== 'Linux':
            codec = 'h264'
        else:
            codec = 'h264_nvenc'
        outfileformat = "{}_output_"+str(postfix)+".mp4" if postfix else "{}_output.mp4"
        outfilename = outfileformat.format(filename[:-4])
        mystr = '{} -i "{}" -vf crop={}:{}:{}:{} -r {} {} -c:v {} "{}" -y'.format(
            ffmpeg_path, filename, whxy[0], whxy[1], whxy[2], whxy[3], fps10, ffmpeg_args, codec, outfilename)
        print(mystr)
        out = os.system(mystr)

if __name__ == '__main__':
    n = len(sys.argv)
    if n == 1:
        folder = cg.uigetfolder()
        if folder == None:
            exit()
        else:
            sys.argv.append(folder)
            
    print(sys.argv[1:])

    video_foldname = os.path.abspath(sys.argv[1])
    config = cg.getfoldconfigpy(video_foldname)
    assert hasattr(config, 'crop_xywh'), 'Need [crop_xywh] in [config_video.py]'
    keepXeqY = getattr(config, 'keepXeqY', True)
    crop_tbg = getattr(config, 'crop_tbg', None)
    crop_tdur = getattr(config, 'crop_tdur', None)
    if type(config.crop_xywh[0]) == list:
        for i, crop_xywh in enumerate(config.crop_xywh):
            whxy = xywh2whxy(crop_xywh, keepXeqY)
            convert_folder_to_mp4(video_foldname, whxy, i+1)
    else:
        whxy = xywh2whxy(config.crop_xywh, keepXeqY)
        convert_folder_to_mp4(video_foldname, whxy)
    print("Succeed")
    
