# python -m lilab.multiview_scripts.dlcRat_2_mat /AB/DLC/
# %% import
import glob
import re
import argparse
import pandas as pd
import numpy as np
import scipy.io as sio


# %% data load
folder_csv = "/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_30fps/whiteblack/15-37-42-white-copy/dlc/"

n_kpt = 14
likelihood_thr = 0.90
# %%
def a1_load_dlc_csv(file):
    with open(file, "r") as f:
        for iline, line in enumerate(f):
            if line.startswith("coords"):
                break
        else:
            raise "Not Valid deeplabcut csv"
    df = pd.read_csv(file, header=list(range(iline)), skiprows=1)
    return df


def a2_extract_points(df):
    arr = df
    for i in range(4):
        if "x" in arr and "y" in arr:
            break
        else:
            arr = arr.droplevel(level=0, axis=1)
    else:
        raise "Not Valid deeplabcut csv"
    arr = arr.drop(columns=["coords"])
    arr_np = arr.to_numpy()
    arr_xyp = arr_np.reshape((arr_np.shape[0], -1, 3))
    return arr_xyp


def a3_thr_filter(arr_xyp, thr=likelihood_thr):
    p_1v = arr_xyp[:, :, 2]
    arr_xy = arr_xyp[:, :, :2]
    idx_bad = p_1v < thr
    arr_xy[idx_bad] = np.nan
    return arr_xy


def convert(folder_csv):
    files_csv = glob.glob(folder_csv + "*.csv")
    fun_re = lambda fn: re.sub("DLC.*csv", "", fn)
    files_out_template = list(map(fun_re, files_csv))

    # convert each file
    for file_csv, file_out_template in zip(files_csv, files_out_template):
        df = a1_load_dlc_csv(file_csv)
        arr_xyp = a2_extract_points(df)
        arr_xy = a3_thr_filter(arr_xyp)
        n_sample, n_kptall, _ = arr_xy.shape
        assert n_kptall / n_kpt in (1.0, 2.0)
        if n_kptall / n_kpt == 1:
            file_out = file_out_template + ".mat"
            sio.savemat(file_out, {"points_2d": arr_xy})
        elif n_kptall / n_kpt == 2:
            file_out = file_out_template + "-black.mat"
            sio.savemat(file_out, {"points_2d": arr_xy[:, :n_kpt, :]})
            file_out = file_out_template + "-white.mat"
            sio.savemat(file_out, {"points_2d": arr_xy[:, n_kpt:, :]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_csv", type=str, default=folder_csv)
    args = parser.parse_args()
    convert(args.folder_csv)
