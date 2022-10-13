# ls /home/jiaming/data/dannce/dannce_predict/*_dannce_predict.pkl | xargs -n 1 -P 4 python scripts_cxf/videopkl2matcalibpkl.py
# %%
import pickle
import numpy as np
from lilab.multiview_scripts_new.s4_matpkl2matcalibpkl import project_points_short
import os.path as osp
import scipy.io as sio
import argparse
videopkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/2021-11-02-bwrat_800x600/30fps/2021-11-02_15-44-53_dannce_predict.pkl'


# %%
def thresh_outlier(key_xyz, p_max, thr=10):
    thr_value = np.percentile(p_max, thr, axis=0, keepdims=True)
    p_max_outlier = p_max < thr_value
    key_xyz[p_max_outlier] = np.nan
    return key_xyz

# %%
def matlab_pose_to_cv2_pose(camParamsOrig):
    keys = ['K', 'RDistort', 'TDistort', 't', 'r']
    camParams = list()
    for icam in range(len(camParamsOrig)):
        camParam = {key: camParamsOrig[icam][key] for key in keys}
        camParam['R'] = camParam['r']
        camParams.append(camParam)

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

def convert(videopkl):
    # %% load data
    pkldata = pickle.load(open(videopkl, 'rb'))
    com_3D = pkldata['com_3D']
    data_3D = pkldata['data_3D']
    # pts3d = data_3D.reshape((data_3D.shape[0], -1, 3)) # (N, K, 3)
    pts3d = np.repeat(com_3D[:, None, :], data_3D.shape[1]//3, axis=1) # (N, K, 3)
    p_max = pkldata['p_max'] # (N, K)
    # pts3d = thresh_outlier(pts3d, p_max)
    N, K, _ = pts3d.shape
    keypoints_xyz_ba = pts3d.reshape(N//2, 2, K, 3)  # (nTime, 2, n_markers, 3)
    keypoints_xyz_pmax = p_max.reshape(N//2, 2, K)   #(nTime, 2, n_markers)

    camParams = pkldata['metadata']['camParams']
    poses = matlab_pose_to_cv2_pose(camParams)
    views = list(poses.keys())

    # %% 3D points project to 2D views
    landmarks_3d = keypoints_xyz_ba.reshape((-1, 3))
    p2d = project_points_short(views, poses, landmarks_3d)
    keypoints_xy_ba = p2d.reshape((len(views), *keypoints_xyz_ba.shape[:-1], 2))

    # %% save pickle
    assert '_dannce_predict.pkl' in videopkl
    videofile = videopkl.replace('_dannce_predict.pkl', '.mp4')
    matcalibpkl = osp.splitext(videofile)[0] + '.matcalibpkl'
    outdict = {'keypoints_xyz_ba': keypoints_xyz_ba,
            'info': {'fps': 30, 'vfile':videofile},
            'keypoints_xy_ba': keypoints_xy_ba,
            'ba_poses': poses}
    pickle.dump(outdict, open(matcalibpkl, 'wb'))

    # %% save mat
    outdict['points_3d'] = keypoints_xyz_ba[:,0,...]
    matcalibmat = osp.splitext(videofile)[0] + '_black.mat'
    sio.savemat(matcalibmat, outdict)
    outdict['points_3d'] = keypoints_xyz_ba[:,1,...]
    matcalibmat = osp.splitext(videofile)[0] + '_white.mat'
    sio.savemat(matcalibmat, outdict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('videopkl', type=str)
    args = parser.parse_args()
    convert(args.videopkl)
    print('done')
