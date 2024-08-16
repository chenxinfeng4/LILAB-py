# python lilab.multiview_scripts.mmposepkl_2mat E:/cxf/mmpose_rat
# %%
import pickle
import os
import glob
import numpy as np
import scipy.io as sio
import argparse
from lilab.multiview_scripts.dlcRat_2_mat import a3_thr_filter

folder_pkl = "/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_30fps/whiteblack/15-37-42-black/"

n_kpt = 14
likelihood_thr = 0.4

# %% functions
def a1_load_mmpose_pkl(pkl):
    with open(pkl, "rb") as f:
        data = pickle.load(f)
        for keypoints in data:
            assert keypoints.shape == (n_kpt, 3)
    arr_xyp = np.array(data)
    return arr_xyp


def convert(folder_pkl):
    files_pkl = glob.glob(os.path.join(folder_pkl, "*.pkl"))
    files_mat = [f.replace(".pkl", ".mat") for f in files_pkl]
    assert files_pkl

    # convert each file
    for pkl, mat in zip(files_pkl, files_mat):
        arr_xyp = a1_load_mmpose_pkl(pkl)
        arr_xy = a3_thr_filter(arr_xyp, likelihood_thr)
        sio.savemat(mat, {"points_2d": arr_xy})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_pkl", default=folder_pkl, nargs="?")
    args = parser.parse_args()
    convert(args.folder_pkl)
