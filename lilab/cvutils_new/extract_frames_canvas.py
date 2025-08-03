# python -m lilab.cvutils_new.extract_frames_canvas testA.mp4
# ls *.mp4 | xargs -n 1 -P 8 python -m lilab.cvutils_new.extract_frames_canvas
import argparse
import os
import cv2
import numpy as np
import tqdm
import os.path as osp
import sys
import glob
from lilab.mmdet_dev.canvas_reader import CanvasReader, CanvasReaderCV

numframe_to_extract = 10
maxlength = 10000
frame_dir = "outframes"
frame_min_interval = 100
classnames = ["ratblack", "ratwhite"]


def extract(video_input, numframe_to_extract, maxlength):
    cap = CanvasReader(video_input)
    length = cap.count
    length = min([maxlength, length - 1])
    downsample_length = length // frame_min_interval
    np.random.seed(0)
    idxframe_to_extract = set(
        np.random.permutation(downsample_length)[:numframe_to_extract]
        * frame_min_interval
        // 3
        * 3
    )
    cap.release()
    ready_to_extract(video_input, idxframe_to_extract)


def ready_to_extract(
    video_input, idxframe_to_extract, rat_name_id=None, outdirname=None
):
    idxframe_max = max(idxframe_to_extract)
    dirname, filename = os.path.split(video_input)
    nakefilename = os.path.splitext(filename)[0]
    if outdirname is None:
        outdirname = os.path.join(dirname, frame_dir)
    os.makedirs(outdirname, exist_ok=True)
    cap = CanvasReader(video_input)
    length = idxframe_max + 1
    rat_name_ids = [rat_name_id] if rat_name_id is not None else range(len(classnames))
    for iframe in tqdm.tqdm(range(length)):
        ret, frame = cap.read()
        if not ret:
            break
        if iframe > idxframe_max:
            break
        if iframe not in idxframe_to_extract:
            continue
        frame_iclass = cap.read_canvas_mask_img_out(img=frame)

        for iclass in rat_name_ids:
            classname = classnames[iclass]
            frame_out = frame_iclass[iclass]
            filename = os.path.join(
                outdirname, nakefilename + "_{}_{:06}.jpg".format(classname, iframe)
            )
            cv2.imwrite(filename, frame_out, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cap.release()


def ready_to_extract_cv(video_input, idxframe_to_extract, rat_name_id=None):
    # sort the idxframe_to_extract
    idxframe_to_extract = sorted(idxframe_to_extract)
    dirname, filename = os.path.split(video_input)
    nakefilename = os.path.splitext(filename)[0]
    os.makedirs(os.path.join(dirname, frame_dir), exist_ok=True)
    cap = CanvasReaderCV(video_input)
    rat_name_ids = [rat_name_id] if rat_name_id is not None else range(len(classnames))
    for iframe in tqdm.tqdm(idxframe_to_extract):
        frame_iclass = cap[iframe]

        for iclass in rat_name_ids:
            classname = classnames[iclass]
            frame_out = frame_iclass[iclass]
            filename = os.path.join(
                dirname,
                frame_dir,
                nakefilename + "_{}_{:06}.jpg".format(classname, iframe),
            )
            cv2.imwrite(filename, frame_out, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract image")
    parser.add_argument(
        "video_path", type=str, default=None, help="path to image or folder"
    )
    args = parser.parse_args()

    video_path = args.video_path
    assert osp.exists(video_path), "video_path not exists"
    if osp.isfile(video_path):
        video_path = [video_path]
    elif osp.isdir(video_path):
        video_path = [
            f
            for f in glob.glob(osp.join(video_path, "*.mp4"))
            if f[-4] not in "0123456789"
        ]
        assert len(video_path) > 0, "no video found"
    else:
        raise ValueError("video_path is not a file or folder")

    # read config_video.py
    for video_input in video_path:
        extract(video_input, numframe_to_extract, maxlength)

    print("Succeed")
