# python -m lilab.dannce.s5_recognize_outcage A/B/C
# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lilab.comm_signal.plotHz import plotHz
from lilab.comm_signal.detectTTL import detectTTL
from lilab.comm_signal.barpatch import barpatch
import scipy
import scipy.signal
import os.path as osp
import argparse
import pandas as pd
import glob

pklfile = "/mnt/liying.cibr.ac.cn_Data_Temp/LS_NAC_fiberphotometry/videomerge/2023-03-27_15-06-43.matcalibpkl"
thr = 0.3


def each_file(pklfile, ifshow=False):
    pkldata = pickle.load(open(pklfile, "rb"))

    # %%
    kpt_pval = pkldata["extra"]["keypoints_xyz_pmax"]
    kpt_pval_m = np.nanmean(kpt_pval, axis=-1)
    Fs = 30

    # filter the kpt_pval_m by a moving average filter
    kpt_pval_s = scipy.signal.medfilt(kpt_pval_m, (11, 1))
    kpt_pval_min = np.min(kpt_pval_s, axis=-1)

    kpt_pval_bool = kpt_pval_min > thr
    assert np.any(kpt_pval_bool)
    trise, tdur = detectTTL(
        kpt_pval_bool, adjacent_type="down-up", adjacent_value=1, fs=Fs
    )
    ind_novalid = tdur < 10
    trise = trise[~ind_novalid]
    tdur = tdur[~ind_novalid]
    tend = trise + tdur
    trise[1:] += 2
    tend[:-1] -= 4
    tdur = tend - trise

    if ifshow:
        barpatch(trise, tdur, HL=[0, 1], color="r")
        plotHz(Fs, kpt_pval_bool)
        plt.gca().autoscale(enable=True, axis="x", tight=True)
        plt.xlabel("Time (s)")

    if np.all(kpt_pval_bool):
        return [[], []]

    epoch_start_l = (trise * Fs).astype(int)
    epoch_end_l = (tend * Fs).astype(int)
    return epoch_start_l, epoch_end_l


def main(pklfile_l):
    vnakename_l = [
        osp.splitext(osp.basename(pklfile))[0] + ".mp4" for pklfile in pklfile_l
    ]
    outcsv = osp.join(osp.dirname(pklfile_l[0]), "video_epoch2.csv")
    df = pd.DataFrame(columns=["VideoPath", "epoch_start", "epoch_end"])
    for (vnakename, pklfile) in zip(vnakename_l, pklfile_l):
        epoch_start_l, epoch_end_l = each_file(pklfile)
        vnakename_repeat = [vnakename] * len(epoch_start_l)
        df = df.append(
            pd.DataFrame(
                {
                    "VideoPath": vnakename_repeat,
                    "epoch_start": epoch_start_l,
                    "epoch_end": epoch_end_l,
                }
            ),
            ignore_index=True,
        )

    df.to_csv(outcsv, index=False)
    print(f"Save to {outcsv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pklfile_or_dir", help="pklfile list")
    args = parser.parse_args()
    pklfile_or_dir = args.pklfile_or_dir

    if osp.isdir(pklfile_or_dir):
        pklfile_l = glob.glob(osp.join(pklfile_or_dir, "*.matcalibpkl"))
        pklfile_l = [p for p in pklfile_l if "smoothed" not in p]
    elif osp.isfile(pklfile_or_dir):
        pklfile_l = [pklfile_or_dir]
    else:
        raise ValueError(f"pklfile_or_dir {pklfile_or_dir} is not a file or dir")
    main(pklfile_l)
