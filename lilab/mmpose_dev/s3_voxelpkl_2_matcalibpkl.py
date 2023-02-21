# python -m lilab.mmpose.s3_voxelpkl_2_matcalibpkl xx
# %%
import pickle
import argparse
import numpy as np
import os.path as osp
from lilab.multiview_scripts_dev.s4_matpkl2matcalibpkl import project_points_short
import scipy.io as sio

black_voxpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/clips/outframes2/2022-04-25_15-44-04_bwt_wwt_02time_ratblack.voxpkl'
white_voxpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/clips/outframes2/2022-04-25_15-44-04_bwt_wwt_02time_ratwhite.voxpkl'

calib_matfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/clips/2022-04-25ball.calibpkl.mat'
calibpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/2022-04-25ball.calibpkl'

def matlab_pose_to_cv2_pose(camParamsOrig):
    keys = ['K', 'RDistort', 'TDistort', 't', 'r']
    camParams = list()
    if type(camParamsOrig) is np.ndarray:
        camParamsOrig = np.squeeze(camParamsOrig)
        for icam in range(len(camParamsOrig)):
            camParam = {key: camParamsOrig[icam][0][key][0] for key in keys}
            camParam['R'] = camParam['r']
            camParams.append(camParam)
    elif type(camParamsOrig) is list:
        for icam in range(len(camParamsOrig)):
            camParam = {key: camParamsOrig[icam][key] for key in keys}
            camParam['R'] = camParam['r']
            camParams.append(camParam)
    else:
        assert len(camParamsOrig) > 2
        for camName, camParamOrg in camParamsOrig.items():
            camParamOrg['R'] = camParamOrg['r']
            camParams.append(camParamOrg)

    # from matlab to opencv
    poses = dict()
    for icam in range(len(camParams)):
        rd = camParams[icam]['RDistort'].reshape((-1))
        td = camParams[icam]['TDistort'].reshape((-1))
        dist = np.zeros((8,))
        dist[:2] = rd[:2]
        dist[2:4] = td[:2]
        dist[3]  = rd[2]
        poses[icam] = {'R': camParams[icam]['R'].T, 
                    't': camParams[icam]['t'].reshape((3,)),
                    'K': camParams[icam]['K'].T - [[0, 0, 1], [0,0,1], [0,0,0]],
                    'dist':dist}
    return poses


def matlab_pose_to_cv2_pose_new(pose):
    keys = ['K', 'dist', 'R', 't']
    outpose = {}
    for k, v in pose.items():
        outpose[k] = {}
        for key in keys:
            outpose[k][key] = np.squeeze(pose[k][key][0][0])

    return outpose


def load_pose_from_mat(matfile):
    mat = sio.loadmat(matfile)
    pose = np.squeeze(mat['ba_poses'])
    pose = {i:pose[i] for i in range(len(pose))}
    pose = matlab_pose_to_cv2_pose_new(pose)
    return pose

def load_pose_from_pkl(calibpkl):
    calib_mat = pickle.load(open(calibpkl, 'rb'))
    pose = calib_mat['ba_poses']
    return pose


def anno_to_pose(annofile):
    annomat = sio.loadmat(annofile)
    camParams = annomat['camParams']
    poses = matlab_pose_to_cv2_pose(camParams)
    return poses

def anno_to_pose_file(annofile):
    ba_poses = anno_to_pose(annofile)
    outcalibfile = osp.splitext(annofile)[0] + '.calibpkl'
    pickle.dump({'ba_poses':ba_poses}, open(outcalibfile, 'wb'))

    from lilab.dannce.s1_ball2mat import convert
    convert(outcalibfile)

    

# %%
