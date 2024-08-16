# python -m lilab.multiview_scripts.mmposepkl_merge2_mat E:/cxf/mmpose_rat/A.mp4
# %%
import pickle
import os.path as osp
import scipy.io as sio
import glob
from ffmpegcv.video_info import get_info
import re
import numpy as np
from lilab.cameras_setup import get_view_xywh_1280x800x10 as get_view_xywh
import argparse

# %%
vfile = "/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/ball/2022-04-29_17-58-45_ball.mp4"


def convert(vfile):
    vinfo = get_info(vfile)

    pkl_files = glob.glob(osp.splitext(vfile)[0] + "_*.pkl")
    p = re.compile(".*(\d+)\.pkl$")
    views = [int(p.findall(pkl_file)[0]) for pkl_file in pkl_files]
    assert len(views) == max(views) + 1 and min(views) == 0

    outdata = {
        "info": {
            "vfile": vfile,
            "nview": len(views),
            "fps": vinfo.fps,
            "vinfo": vinfo._asdict(),
        },
        "views_xywh": get_view_xywh(),
        "keypoints": {},
    }

    keypoints = [[] for _ in range(len(views))]

    for view, pkl_file in zip(views, pkl_files):
        data = pickle.load(open(pkl_file, "rb"))
        assert isinstance(data, dict) and str(view) in data
        keypoints[view] = data[str(view)]

    keypoints = np.array(keypoints)
    outdata["keypoints"] = keypoints

    # %% save to mat file
    # outmat = osp.splitext(vfile)[0] + '.mat'
    outpkl = osp.splitext(vfile)[0] + ".matpkl"
    # sio.savemat(outmat, outdata)
    pickle.dump(outdata, open(outpkl, "wb"))
    print("saved to", outpkl)
    return outpkl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vfile", type=str)
    args = parser.parse_args()
    convert(args.vfile)
