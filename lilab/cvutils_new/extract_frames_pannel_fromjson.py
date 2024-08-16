# python -m lilab.cvutils_new.extract_frames_pannel_fromjson out.json
import argparse
import os
import cv2
import numpy as np
import tqdm
import os
import os.path as osp
import json
import cv2
from lilab.cameras_setup import get_view_xywh_wrapper
import ffmpegcv

crop_xywhs = get_view_xywh_wrapper(9)


def parser_json(json_file, dir_name=None):
    data = json.load(open(json_file, "r"))
    keys = list(data.keys())
    vfiles = [k[:-2] for k in keys]
    pannels = [k[-1] for k in keys]
    iframealls = [data[k] for k in keys]
    if dir_name is None:
        dir_name = osp.dirname(osp.dirname(json_file))
    for vfile, pannel, iframeall in zip(
        tqdm.tqdm(vfiles, position=0), pannels, iframealls
    ):
        full_vfile = os.path.join(dir_name, vfile)
        assert osp.exists(full_vfile), "video_path not exists"
        vid = ffmpegcv.VideoCaptureNV(
            full_vfile, gpu=0, crop_xywh=crop_xywhs[int(pannel)]
        )
        # make dir for each output pannel
        out_dir = osp.join(
            osp.dirname(full_vfile),
            "{}_pannel_{}".format(osp.basename(full_vfile), pannel),
        )
        os.makedirs(out_dir, exist_ok=True)
        iframeall = [int(iframe) for iframe in iframeall]
        iframemax = max(iframeall)
        with vid:
            for iframe, frame in enumerate(
                tqdm.tqdm(vid, position=1, desc=f"{pannel}")
            ):
                if iframe in iframeall:
                    out_file = osp.join(out_dir, "{}.jpg".format(iframe))
                    cv2.imwrite(out_file, frame)
                if iframe == iframemax:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json", type=str)
    parser.add_argument("--dir_name", type=str, default=None)
    args = parser.parse_args()
    parser_json(args.json, args.dir_name)
    print("Done")
