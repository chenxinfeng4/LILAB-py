# python -m lilab.cvutils_new.extract_frames_fromjson out.json
#%%
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
from collections import OrderedDict
from lilab.paralleltool.gpuquery_api import GpuQuery, ngpu
from multiprocessing import Pool

pannels_xywh = get_view_xywh_wrapper(9)
frame_dir = "outframes_raw_2"

csv_file = "/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/shank3HetxWT_PND75/SHANK3_rat_info_treat_info_mask有误.csv"
dir_name = (
    "/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/shank3HetxWT_PND75/batch2/"
)

df = pd.read_csv(csv_file)
assert {"video_nake", "iframe", "ipannel"} <= set(df.columns)
df.sort_values(by=["video_nake", "ipannel", "iframe"], inplace=True)

video_nake_l = df["video_nake"].unique()
video_file_l = [osp.join(dir_name, f + ".mp4") for f in video_nake_l]
video_nake_file_dict = dict(zip(video_nake_l, video_file_l))
assert all([osp.isfile(f) for f in video_file_l])

df_groupby = df.groupby(["video_nake", "ipannel"])["iframe"].apply(list).reset_index()

task_args = []
for video_nake, ipannel, idxframe_to_extract in zip(
    df_groupby["video_nake"], df_groupby["ipannel"], df_groupby["iframe"]
):
    video_input = video_nake_file_dict[video_nake]
    # ready_to_extract(video_input, idxframe_to_extract, dir_name, ipannel)
    task_args.append((video_input, idxframe_to_extract, dir_name, ipannel))

pool = Pool(n=ngpu)
for result in pool.starmap(ready_to_extract, task_args):
    print(f"Got result: {result}", flush=True)


def ready_to_extract(video_input, idxframe_to_extract, dir_name, ipannel):
    idxframe_max = max(idxframe_to_extract)
    _, filename = os.path.split(video_input)
    nakefilename = os.path.splitext(filename)[0]
    os.makedirs(os.path.join(dir_name, frame_dir), exist_ok=True)
    x, y, w, h = pannels_xywh[ipannel]
    gpu = GpuQuery().get()
    cap = ffmpegcv.VideoCaptureNV(
        video_input, gpu=gpu, crop_xywh=pannels_xywh[ipannel], pix_fmt="gray"
    )
    length = idxframe_max + 1
    for iframe in tqdm.tqdm(range(length), position=gpu):
        ret, imgDataCrop = cap.read()
        if not ret:
            break
        if iframe > idxframe_max:
            break
        if iframe not in idxframe_to_extract:
            continue
        frame_name = nakefilename + "_" + str(iframe) + ".jpg"
        filename = os.path.join(
            dir_name,
            frame_dir,
            nakefilename + "_{:1}_{:06}.jpg".format(ipannel, iframe),
        )
        cv2.imwrite(filename, imgDataCrop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str)
    parser.add_argument("--dir_name", type=str, default=None)
    args = parser.parse_args()
    parser_json(args.csv_file, args.dir_name)
    print("Done")
