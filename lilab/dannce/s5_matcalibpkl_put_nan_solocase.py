# python -m lilab.dannce.s5_matcalibpkl_put_nan_solocase A/B/SOLO.matcalibpkl
import argparse
import os.path as osp
import pickle
import numpy as np

file = "/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/voxelmat_D36_SOLO/2023-01-02_14-06-45BwVPA.smoothed_foot.matcalibpkl"

p_thr = 0.1  # below is considered nan


def convert(file):
    s = pickle.load(open(file, "rb"))

    p_max = s["extra"]["keypoints_xyz_pmax"]
    p_max_median = np.median(p_max, axis=[0, -1])
    assert p_max_median.min() < 0.1 and p_max_median.max() > 0.8

    index_nan = np.where(p_max_median < p_thr)[0].tolist()

    for ianimal in index_nan:
        s["keypoints_xyz_ba"][:, ianimal] = np.nan
        s["keypoints_xy_ba"][:, :, ianimal] = np.nan

    if len(index_nan):
        pickle.dump(s, open(file, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Put nan to matcalibpkl when solo rat case."
    )
    parser.add_argument(
        "file_or_dir", type=str, help="the input pkl file or directory."
    )
    args = parser.parse_args()

    assert args.file_or_dir, "None input"
    if osp.isfile(args.file_or_dir):
        convert(args.file_or_dir)
    elif osp.isdir(args.file_or_dir):
        import glob

        pkl_list = glob.glob(osp.join(args.file_or_dir, "*.matcalibpkl"))
        for pkl in pkl_list:
            convert(pkl)
    else:
        assert False, "Wrong input {}".format(args.file_or_dir)
