# python -m lilab.cvutils_new.reverse_canvas_video /mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/clips/outframes
import argparse
import os
import cv2
import numpy as np
import tqdm
import os.path as osp
import sys
import glob

from lilab.cameras_setup import get_view_xywh_1280x800x10 as get_view_xywh
from lilab.cameras_setup import get_view_hflip
import ffmpegcv


def convert(filename):
    views = get_view_xywh()
    hflips = get_view_hflip()
    filename_out = osp.splitext(filename)[0] + "_reverse.mp4"
    views = [views[i] for i in range(len(views)) if hflips[i]]
    if len(views) == 0:
        return
    vidin = ffmpegcv.VideoCaptureNV(filename)
    vidout = ffmpegcv.VideoWriter(filename_out)
    for img in tqdm.tqdm(vidin):
        img = img.copy()
        for view in views:
            x, y, w, h = view
            im1 = img[y : y + h, x : x + w]
            img[y : y + h, x : x + w] = im1[:, ::-1]
        vidout.write(img)
    vidin.release()
    vidout.release()
    print("Video {} was converted".format(filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_path", type=str, default=None, help="path to image or folder"
    )

    args = parser.parse_args()

    image_path = args.image_path
    assert osp.exists(image_path), "image_path not exists"
    if osp.isfile(image_path):
        image_path = [image_path]
    elif osp.isdir(image_path):
        # image_path = [f for f in glob.glob(osp.join(image_path, '*.jpg'))
        #                 if f[-4] not in '0123456789']
        image_path = glob.glob(osp.join(image_path, "*.mp4"))
        assert len(image_path) > 0, "no image found"
    else:
        raise ValueError("image_path is not a file or folder")

    for filename in image_path:
        convert(filename)
    print("Succeed")
