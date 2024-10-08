#!/usr/bin/python
# !pyinstaller -F extract_frames.py -i mypython.ico
# chenxinfeng
# ------使用方法------
# 直接拖动文件夹/视频到EXE中

import os
import cv2
import tqdm
import sys
from glob import glob
from typing import Union

try:
    from . import cxfguilib as cg
except Exception as e:
    import cxfguilib as cg

frame_dir = "outframes"


def timestr_to_seconds(frames_time):
    frames_seconds = []
    for frame_time in frames_time:
        if type(frame_time) == str:
            frame_seconds = sum(
                x * int(t) for x, t in zip([3600, 60, 1], frame_time.split(":"))
            )
        else:
            frame_seconds = frame_time
        frames_seconds.append(frame_seconds)
    return frames_seconds


def extract(video_input, frame_seconds):
    dirname, filename = os.path.split(video_input)
    nakefilename = os.path.splitext(filename)[0]
    cap = cv2.VideoCapture(video_input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(os.path.join(dirname, frame_dir), exist_ok=True)

    for iframe, frame_seconds in enumerate(tqdm.tqdm(frame_seconds)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_seconds * fps))
        ret, frame = cap.read()
        if not ret:
            continue
        filename = os.path.join(
            dirname, frame_dir, nakefilename + "_{0:05}.png".format(iframe)
        )
        cv2.imwrite(filename, frame)

    cap.release()


def extract_crop(
    filename: str, indexby="frame", crop_xywh=None, timestamps=None, ipannel=None
):

    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    imgname = os.path.splitext(filename)[0]
    imgpostfix = f"_output_{ipannel}" if ipannel is not None else ""
    ret, img = cap.read()
    for i, frame_seconds in enumerate(tqdm.tqdm(timestamps)):
        cap.set(cv2.CAP_PROP_POS_MSEC, frame_seconds * 1000)
        ret, img = cap.read()
        if not ret:
            raise Exception("read failed")
        if crop_xywh is not None:
            img = img[
                crop_xywh[1] : crop_xywh[1] + crop_xywh[3],
                crop_xywh[0] : crop_xywh[0] + crop_xywh[2],
            ]
        index = int(frame_seconds * fps) if indexby == "frame" else i + 1
        imgfullname = f"{imgname}_{index:06d}{imgpostfix}.png"
        cv2.imwrite(imgfullname, img)
    cap.release()


if __name__ == "__main__":
    n = len(sys.argv)
    if n == 1:
        folder = input("Choose video folder: >> ")
        if folder == None:
            exit()
        else:
            sys.argv.append(folder)

    print(sys.argv[1:])

    # check input is file or folder
    file_or_folderpath = sys.argv[1]
    if os.path.isfile(file_or_folderpath):
        video_foldname, f = os.path.split(file_or_folderpath)
        os.chdir(video_foldname)
        filenamesList = [f]
    elif os.path.isdir(file_or_folderpath):
        video_foldname = file_or_folderpath
        os.chdir(video_foldname)
        filenamesList = glob(r"*.avi") + glob(r"*.mp4") + glob(r"*.mkv")
        assert len(filenamesList), "Folder contain no AVI/MP4/MKV videos!"
    else:
        assert False, "Input should be FILE or FOLDER"

    # read config_video.py
    config = cg.getfoldconfigpy(".")
    frames_time = getattr(config, "frames_time", None)
    frames_seconds = timestr_to_seconds(frames_time)

    for video_input in filenamesList:
        extract(video_input, frames_seconds)

    print("Succeed")
