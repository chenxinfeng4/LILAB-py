# python -m lilab.dannce.s7_matcalibpkl_to_BeA A/B/C.matcalibpkl
# %%
"""
将 dannce 的matcalibpkl 文件转为 蔚鹏飞 BeA 的matlab 文件
[1] Huang, Kang, et al. "A hierarchical 3D-motion learning framework for animal 
spontaneous behavior mapping." Nature communications 12.1 (2021): 2784.
"""
import argparse
import os
import pickle

import numpy as np
import scipy.io as sio

CenIndex = 5  # back
VAIndex = 6  # tail
SDSDimens = [16, 17, 18]  # tail

matcalibpkl = "/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-25_16-35-53_SHANK20_wKOxbHet.smoothed_foot.matcalibpkl"


def convert(matcalibpkl):
    with open(matcalibpkl, "rb") as f:
        matcalib = pickle.load(f)

    # %%
    kpt3d = matcalib["keypoints_xyz_ba"]
    frame_max = 30 * 60 * 10  # 10min
    kpt3d = kpt3d[:frame_max]
    Nframe, Naminal, Njoint, Ndim = kpt3d.shape

    # %%
    kpt3d_flat = kpt3d.reshape(Nframe, Naminal, -1)
    kpt3d_flat2 = np.concatenate([kpt3d_flat[:, i] for i in range(Naminal)])
    assert kpt3d_flat2.shape == (Nframe * Naminal, Njoint * Ndim)

    outdict = {
        "coords3d": kpt3d_flat2.astype(np.float64),
        # "CenIndex": CenIndex,
        # "VAIndex": VAIndex,
        # "SDSDimens": SDSDimens,
    }

    outfile = os.path.splitext(matcalibpkl)[0] + "_BA.mat"
    sio.savemat(outfile, outdict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("matcalibpkl", type=str)
    args = parser.parse_args()

    assert os.path.isfile(args.matcalibpkl)
    convert(args.matcalibpkl)
