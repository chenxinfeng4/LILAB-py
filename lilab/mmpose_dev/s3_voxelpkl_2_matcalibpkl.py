# python -m lilab.mmpose.s3_voxelpkl_2_matcalibpkl xx
# %%
import pickle
import argparse
import numpy as np
from lilab.multiview_scripts_new.s4_matpkl2matcalibpkl import project_points_short
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




# %%
black_mat = pickle.load(open(black_voxpkl, 'rb'))
white_mat = pickle.load(open(white_voxpkl, 'rb'))

# %%
black_xyz_baglobal = black_mat['keypoints_xyz_baglobal']
white_xyz_baglobal = white_mat['keypoints_xyz_baglobal']
pose = load_pose_from_mat(calib_matfile)

# %%
keypoints_xyz_baglobal = np.concatenate((black_xyz_baglobal, white_xyz_baglobal), axis=1)

views = list(pose.keys())
landmarks_3d = keypoints_xyz_baglobal.reshape((-1, 3))
p2d = project_points_short(views, pose, landmarks_3d)
keypoints_xy_ba = p2d.reshape((10,) + keypoints_xyz_baglobal.shape[:-1] + (2,))

# %%
outdict = {'keypoints_xyz_baglobal': keypoints_xyz_baglobal,
         'info': {'fps': 30, 'vfile':'/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/clips/2022-04-25_15-44-04_bwt_wwt_02time.mp4'},
         'keypoints_xy_ba': keypoints_xy_ba,
         'ba_poses': pose}

pickle.dump(outdict, open('/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/clips/2022-04-25_15-44-04_bwt_wwt_02time.matcalibpkl', 'wb'))

# %%
