# python -m lilab.cvutils_new.extract_frames_fromjson out.json
import argparse
import os
import cv2
import numpy as np
import tqdm
import os.path as osp
import json
import ffmpegcv

frame_dir = "outframes_raw"


def parser_json(json_file, dir_name=None):
    data = json.load(open(json_file, "r"))
    vfiles = list(data.keys())
    if dir_name is None:
        dir_name = osp.dirname(osp.dirname(json_file))
    filenames_dict = {
        filename: osp.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(dir_name)
        for filename in filenames
    }
    for vfile in vfiles:
        full_vfile = filenames_dict[vfile]
        # full_vfile = os.path.join(dir_name, vfile)
        assert osp.exists(full_vfile), "video_path not exists"
        idxframes = [int(idxframe) for idxframe in data[vfile]]
        ready_to_extract(full_vfile, idxframes, dir_name)


def ready_to_extract(video_input, idxframe_to_extract, outdirname, isrootdir=False):
    idxframe_max = max(idxframe_to_extract)
    _, filename = os.path.split(video_input)
    nakefilename = os.path.splitext(filename)[0]
    outdirname = outdirname if isrootdir else os.path.join(outdirname, frame_dir)
    os.makedirs(outdirname, exist_ok=True)
    cap = ffmpegcv.VideoCaptureNV(video_input, pix_fmt="nv12")
    length = idxframe_max + 1
    filenames = []
    for iframe in tqdm.tqdm(range(length)):
        ret, frame = cap.read()
        if not ret:
            break
        if iframe > idxframe_max:
            break
        if iframe not in idxframe_to_extract:
            continue

        filename = os.path.join(outdirname, nakefilename + "_{:06}.jpg".format(iframe))
        frame_color = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
        cv2.imwrite(filename, frame_color, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        filenames.append(filename)
    cap.release()
    return filenames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json", type=str)
    parser.add_argument("--dir_name", type=str, default=None)
    args = parser.parse_args()
    parser_json(args.json, args.dir_name)
    print("Done")
