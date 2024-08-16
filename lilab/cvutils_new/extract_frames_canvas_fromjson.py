# python -m lilab.cvutils_new.extract_frames_canvas_fromjson out.json
import argparse
import os
import cv2
import numpy as np
import tqdm
import os.path as osp
import json
from lilab.cvutils_new.extract_frames_canvas import (
    ready_to_extract,
    ready_to_extract_cv,
)

rat_name_map = {
    "1": 1,
    "0": 0,
    "black": 0,
    "white": 1,
    "ratblack": 0,
    "ratwhite": 1,
    "rat_black": 0,
    "rat_white": 1,
}


def parser_json(json_file, dir_name=None, rat_name=None):
    data = json.load(open(json_file, "r"))
    vfiles = list(data.keys())
    rat_name_id = rat_name_map[rat_name] if rat_name is not None else None
    if dir_name is None:
        dir_name = osp.dirname(osp.dirname(json_file))
    outdirname = osp.join(dir_name, "outframes_voxel")
    os.makedirs(outdirname, exist_ok=True)
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
        ready_to_extract(full_vfile, idxframes, rat_name_id, outdirname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json", type=str)
    parser.add_argument("--dir_name", type=str, default=None)
    parser.add_argument("--rat_name", type=str, default=None)
    args = parser.parse_args()
    parser_json(args.json, args.dir_name, args.rat_name)
    print("Done")
