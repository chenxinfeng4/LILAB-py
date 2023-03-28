# python -m lilab.mmdet_dev.s4_segpkl_put_com3d xxx.segpkl xxx.calibpkl
# ls *.segpkl | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s4_segpkl_put_com3d
import argparse
import pickle

import numpy as np
import pycocotools.mask as maskUtils
import scipy.signal
import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage.measurements import center_of_mass

from lilab.multiview_scripts_dev.s6_calibpkl_predict import CalibPredict

segpkl = "/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-25_17-08-41_SHANK20_wHetxbKO.segpkl"

calibpkl = "/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/TPH2-KO-multiview-202201/male/ball/ball.calibpkl"


def ims_to_com2ds(ims):
    coms_2d = []
    for im_mask in ims:
        assert im_mask.ndim == 2
        com_2d = (
            center_of_mass(im_mask)[::-1]
            if np.max(im_mask) >= 1
            else np.ones((2,)) + np.nan
        )
        coms_2d.append(com_2d)
    coms_2d = np.array(coms_2d)
    return coms_2d


# %%
def convert(segpkl, calibpkl):
    pkl_data = pickle.load(open(segpkl, "rb"))
    assert "coms_3d" not in pkl_data

    views_xywh = pkl_data["views_xywh"]
    segdata = pkl_data["segdata"]
    calibPredict = CalibPredict(calibpkl)

    if len(views_xywh) == 6:
        mask_original_shape = (600, 800)
    elif len(views_xywh) == 10:
        mask_original_shape = (800, 1200)
    else:
        raise NotImplementedError

    nviews = len(views_xywh)
    nframes = len(segdata[0])
    nclass = 2

    coms_real_2d = np.zeros((nviews, nframes, nclass, 2))
    for iframe in tqdm.tqdm(range(nframes)):
        mask_ims = [[] for _ in range(nclass)]
        for iview in range(nviews):
            segdata_iview = segdata[iview][iframe]
            for iclass in range(nclass):
                mask = segdata_iview[1][iclass]
                mask = maskUtils.decode(mask)[:, :, 0]
                mask_ims[iclass].append(mask)
        for iclass in range(nclass):
            coms_real_2d[:, iframe, iclass, :] = ims_to_com2ds(mask_ims[iclass])

    #%%
    mask_real_shape = mask.shape
    resize_scale = np.array(
        [
            [
                mask_original_shape[1] / mask_real_shape[1],
                mask_original_shape[0] / mask_real_shape[0],
            ]
        ]
    )
    coms_2d = coms_real_2d * resize_scale

    coms_3d = calibPredict.p2d_to_p3d(coms_2d)  # nsample_nclass_3

    #%%
    coms_3d_smooth = np.zeros_like(coms_3d)
    for is1 in range(coms_3d.shape[1]):
        for is2 in range(coms_3d.shape[2]):
            temp = scipy.signal.medfilt(coms_3d[:, is1, is2], kernel_size=7)
            temp = gaussian_filter1d(temp, sigma=5)
            coms_3d_smooth[:, is1, is2] = temp

    coms_2d_smooth = calibPredict.p3d_to_p2d(coms_3d_smooth)  # nview_nsample_nclass_2
    pkl_data.update(
        {
            "coms_3d": coms_3d_smooth,
            "coms_2d": coms_2d_smooth,
            "ba_poses": calibPredict.poses,
        }
    )

    # %% dump
    pickle.dump(pkl_data, open(segpkl, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("segpkl", type=str)
    parser.add_argument("--calibpkl", type=str, default=calibpkl)
    args = parser.parse_args()
    convert(args.segpkl, calibpkl)
