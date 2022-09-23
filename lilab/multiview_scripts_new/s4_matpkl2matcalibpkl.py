# python -m lilab.multiview_scripts_new.s4_matpkl2matcalibpkl  xxx.matpkl xxx.calibpkl 
# %%
import os.path as osp
import pickle
import numpy as np
from multiview_calib.bundle_adjustment_scipy import  (pack_camera_params,
        unpack_camera_params, undistort_points, triangulate, project_points)
import itertools
import argparse

# %%
matpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/ball/2022-04-29_17-58-45_ball.matpkl'
calibpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/ball/2022-04-29_17-58-45_ball.calibpkl'

# %%
def triangulate_all_pairs_short(views, landmarks, camera_params):
    
    n_cameras = len(views)

    # to speed things up
    poses = []
    landmarks_undist_withnan = {}
    for j in range(n_cameras):
        K,R,t,dist = unpack_camera_params(camera_params[j])
        poses.append((K,R,t,dist))
        points = landmarks[views[j]]
        landmarks_undist_withnan[views[j]] = undistort_points(points, K, dist)
    

    p3d_allview_withnan = []
    for j1, j2 in itertools.combinations(range(n_cameras), 2):
        K1,R1,t1,dist1 = poses[j1]
        K2,R2,t2,dist2 = poses[j2]
        pts1 = landmarks_undist_withnan[views[j1]]
        pts2 = landmarks_undist_withnan[views[j2]]
        p3d = triangulate(pts1, pts2, K1, R1, t1, None, K2, R2, t2, None)
        p3d_allview_withnan.append(p3d)
    p3d_allview_withnan = np.array(p3d_allview_withnan) #nviewpairs_nsample_xyz
        
    return p3d_allview_withnan


def build_input_short(views, poses, landmarks):
    # camera parameters
    camera_params = [pack_camera_params(poses[view]['K'], poses[view]['R'],
                                        poses[view]['t'], poses[view]['dist']) 
                          for view in views]
    camera_params = np.float64(camera_params)

    # triangulate 3D positions from all possible pair of views
    p3d_allview_withnan = triangulate_all_pairs_short(views, landmarks, camera_params)
    p3d = np.nanmedian(p3d_allview_withnan, axis=0) #nsample_xyz
    return p3d


def project_points_short(views, poses, keypoints_xyz, image_shape=None):
    assert keypoints_xyz.shape[-1] == 3
    p3d = keypoints_xyz.reshape((-1, 3))
    if image_shape is None:
        image_shape = (np.inf, np.inf)
    nviews = len(poses)
    p2d = np.empty((nviews, p3d.shape[0], 2)) + np.nan   # nviews_nsample_2
    mask_inside = lambda proj: np.logical_and.reduce([proj[:,0]>0, proj[:,0]<image_shape[1],
                                             proj[:,1]>0, proj[:,1]<image_shape[0]])
    
    for view in views:
        param = lambda NAME : np.array(poses[view][NAME])
        K,R,t,dist = param('K'), param('R'), param('t'), param('dist')
        p2d_tmp, _ = project_points(p3d, K, R, t, dist, image_shape)
        p2d_tmp[~mask_inside(p2d_tmp)] = np.nan
        p2d[view] = p2d_tmp

    keypoints_xy = p2d.reshape((len(views), *keypoints_xyz.shape[:-1], 2))
    return keypoints_xy


def compare_error(keypoints_xy, keypoints_xy_ba):
    keypoints_xy_ba_error = keypoints_xy - keypoints_xy_ba
    error = np.sqrt(np.sum(keypoints_xy_ba_error**2, axis=3))
    error = error.reshape(len(keypoints_xy), -1)
    error_median = np.nanmedian(error, axis=1).round(1)
    return error_median

# %%
def convert(matpkl, calibpkl):
    mat = pickle.load(open(matpkl, 'rb'))
    calib = pickle.load(open(calibpkl, 'rb'))
    thr = 0.25
    keypoints = mat['keypoints']
    indmiss = keypoints[..., 2] < thr
    keypoints_xy = keypoints[..., :2]  # (nview, times, [nobj,] nkeypoints, 2)
    keypoints_xy[indmiss] = np.nan
    orginal_shape = keypoints_xy.shape

    landmarks = keypoints_xy.reshape(orginal_shape[0], -1, orginal_shape[-1])
    landmarks = {i: landmarks[i] for i in range(orginal_shape[0])} #{iview: (times * nkeypoints, 2)}

    views = calib['setup']['views']
    image_shape = list(calib['intrinsics'].values())[0]['image_shape'] #HxW


    # %%
    keypoints_xyz_ba = []
    keypoints_xy_ba  = []
    keypoints_xyz_baglobal = []

    # %%
    assert 'ba_poses' in calib
    poses = calib['ba_poses']
    p3d = build_input_short(views, poses, landmarks)   # nsample_xyz
    p2d = project_points_short(views, poses, p3d, image_shape) # nviews_nsample_xy
    keypoints_xyz_ba = p3d.reshape(*orginal_shape[1:-1],3)
    keypoints_xy_ba = p2d.reshape(*orginal_shape[:-1], 2)
    # calculate the error between the keypoints_xy and keypoints_xy_ba
    error_median = compare_error(keypoints_xy, keypoints_xy_ba)
    print('[BA] error_median pixel:', error_median)

    # %%
    if 'ba_global_params' in calib:
        params = calib['ba_global_params']
        fun_ba2global = eval(params['strfun_ba2global'])
        keypoints_xyz_baglobal = fun_ba2global(keypoints_xyz_ba, params['kargs'])
    elif 'ba_global_poses' in calib:
        poses = calib['ba_global_poses']
        p3d = build_input_short(views, poses, landmarks)   # nsample_xyz
        p2d = project_points_short(views, poses, p3d, image_shape) # nviews_nsample_xy
        keypoints_xyz_baglobal = p3d.reshape(orginal_shape[1], orginal_shape[2], 3)

    # %%
    outdict = {'setup': calib['setup'],
            'intrinsics': calib['intrinsics'],
            'ba_poses': calib['ba_poses'],
            'ba_global_params': calib['ba_global_params'],
            'keypoints_xyz_ba': keypoints_xyz_ba,
            'keypoints_xy_ba': keypoints_xy_ba,
            'keypoints_xyz_baglobal': keypoints_xyz_baglobal}

    outdict.update(mat)

    # %% save
    outpkl = osp.splitext(matpkl)[0] + '.matcalibpkl'
    pickle.dump(outdict, open(outpkl, 'wb'))
    print('python -m lilab.multiview_scripts_new.s5_show_calibpkl2video', outpkl)
    return outpkl


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('matpkl', help='mat file')
    parser.add_argument('calibpkl', help='calib file')
    args = parser.parse_args()
    convert(args.matpkl, args.calibpkl)