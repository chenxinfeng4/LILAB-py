# python -m lilab.multiview_scripts_dev.s4_matpkl2matcalibpkl  xxx.matpkl xxx.calibpkl 
# %%
import os.path as osp
import pickle
import numpy as np
from multiview_calib.calibpkl_predict import CalibPredict
import itertools
import argparse

# %%
#matpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/ball/2022-04-29_17-58-45_ball.matpkl'
#calibpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/ball/2022-04-29_17-58-45_ball.calibpkl'
matpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-2-21_marmosettracking/2024-02-21_15-07-36.matpkl'
calibpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-1-31_sync/ballmove/ball_move_cam1.aligncalibpkl'


def compare_error(keypoints_xy, keypoints_xy_ba):
    error = np.linalg.norm(keypoints_xy - keypoints_xy_ba, axis=3).ravel()
    error_median = np.nanmedian(error).round(1)
    return error_median


# %%
def convert(matpkl, calibpkl, fix_camnum=False):
    mat = pickle.load(open(matpkl, 'rb'))
    calib = pickle.load(open(calibpkl, 'rb'))
    thr = 0.25
    keypoints = mat['keypoints']
    indmiss = keypoints[..., 2] < thr
    keypoints_xy = keypoints[..., :2]  # (nview, times, [nobj,] nkeypoints, 2)
    keypoints_xy[indmiss] = np.nan
    orginal_shape = keypoints_xy.shape
    ba_poses = calib['ba_poses']
    nviews_matpkl = len(keypoints_xy)
    nviews_calibpkl = len(ba_poses)
    assert nviews_matpkl <= nviews_calibpkl
    if fix_camnum and nviews_matpkl != nviews_calibpkl:
        print(f'Warning: nviews_matpkl {nviews_matpkl} != nviews_calibpkl {nviews_calibpkl}.'
               f'Only use {min(nviews_matpkl, nviews_calibpkl)} views')
        ba_poses = {i: ba_poses[i] for i in range(min(nviews_matpkl, nviews_calibpkl))}
    calibobj = CalibPredict({'ba_poses': ba_poses})
    keypoints_xyz_ba = calibobj.p2d_to_p3d(keypoints_xy)
    keypoints_xy_ba = calibobj.p3d_to_p2d(keypoints_xyz_ba)

    # calculate the error between the keypoints_xy and keypoints_xy_ba
    error_median = compare_error(keypoints_xy, keypoints_xy_ba)
    print('[BA] error_median pixel:', error_median)

    # %%
    outdict = {
            'ba_poses': ba_poses,
            'keypoints_xyz_ba': keypoints_xyz_ba,
            'keypoints_xy_ba': keypoints_xy_ba}
    outdict.update(mat)

    # %% save
    outpkl = osp.splitext(matpkl)[0] + '.matcalibpkl'
    pickle.dump(outdict, open(outpkl, 'wb'))
    print('python -m lilab.multiview_scripts_dev.s5_show_calibpkl2video', outpkl)
    return outpkl


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('matpkl', help='mat file')
    parser.add_argument('calibpkl', help='calib file')
    parser.add_argument('--fix-camnum', action='store_true', help='if camnum is not equal to nviews, fix it')
    args = parser.parse_args()
    convert(args.matpkl, args.calibpkl, args.fix_camnum)