# ls *matcalibpkl | xargs -n 1 python -m lilab.dannce.s8_final_check
import numpy as np
import pickle
import argparse
import os.path as osp

minlen = 27000
nclass = 2
nview = 9


def check(matcalibpklfile):
    print(f"Checking {osp.basename(matcalibpklfile)}")
    mat = pickle.load(open(matcalibpklfile, "rb"))
    assert {"keypoints_xyz_ba", "keypoints_xy_ba", "ba_poses"} <= mat.keys()
    nview_, nsample_, nclass_, _, _ = mat["keypoints_xy_ba"].shape
    assert nview_ == nview and nclass_ == nclass and nsample_ >= minlen, str(
        (nview_, nsample_, nclass_)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("matcalibpkl", type=str, help="matcalibfile")
    args = parser.parse_args()
    check(args.matcalibpkl)
