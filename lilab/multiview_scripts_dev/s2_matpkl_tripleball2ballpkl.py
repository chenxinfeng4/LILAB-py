# python -m lilab.multiview_scripts_dev.s2_matpkl2ballpkl ../data/matpkl/ball.matpkl --time 1 12 24 35 48
import pickle
import argparse
from lilab.multiview_scripts_dev.s2_matpkl2ballpkl import convert
import lilab.multiview_scripts_dev.s2_matpkl2ballpkl as S2MPBP
import numpy as np
second_based = True
pthr = 0.62
matfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/ball/2022-04-29_17-58-45_ball.matpkl'


def load_mat(matfile):
    data = pickle.load(open(matfile, 'rb'))
    keypoint = data['keypoints'].copy()
    fps = data['info']['fps']
    vfile = data['info']['vfile']
    views_xywh = data['views_xywh']

    assert keypoint.ndim == 4
    nview, nframe, nk, ncoord_xyp = keypoint.shape
    assert ncoord_xyp == 3, "xyp is expected"
    keypoint_flat = np.concatenate([keypoint[:, :, k, :] for k in range(nk)], axis=1) #nview, nframe*nk, ncoord_xyp
    # keypoint_flat = keypoint[:,:,0,:]
    return keypoint_flat, fps, vfile, views_xywh

S2MPBP.load_mat = load_mat

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('matfile', type=str)
    parser.add_argument('--time', type=float, nargs='+')
    parser.add_argument('--force-setupname', type=str, default=None)
    args = parser.parse_args()
    assert len(args.time) == 5, "global_time should be 5 elements"
    convert(args.matfile, args.time, args.force_setupname)
