"""
from lilab.multiview_scripts_dev.s6_calibpkl_predict import CalibPredict
calibPredict = CalibPredict(calibfile)
p2d = calibPredict.p3d_to_p2d(p3d)
p3d = calibPredict.p2d_to_p3d(p2d)
"""
# %% imports
from builtins import NotImplementedError, isinstance
import numpy as np
import pickle
import os
import itertools
from multiview_calib.bundle_adjustment_scipy import (undistort_points, triangulate, project_points)


class CalibPredict:
    def __init__(self, calibfile):
        """
        init by the 'calibfile' as pkl file, or 'calib' as dict.
        """
        if isinstance(calibfile, str):
            assert os.path.isfile(calibfile)
            with open(calibfile, 'rb') as f:
                calib = pickle.load(f)
        elif isinstance(calibfile, dict):
            calib = calibfile
        else:
            raise NotImplementedError
        assert {'ba_poses'} <= set(calib.keys()), 'calib file is not correct'
        self.poses = calib['ba_poses']
        self.image_shape = list(calib['intrinsics'].values())[0]['image_shape'] if 'intrinsics' in calib else None #HxW
        self.views = sorted(list(calib['ba_poses'].keys()))

    def p3d_to_p2d(self, p3d:np.ndarray, image_shape=None) -> np.ndarray:
        """
        Convert 3D points to 2D points.
        ---------- inputs ----------
        p3d: nsample_**_xyz
        image_shape: (H, W)

        ---------- outputs ----------
        p2d: nview_nsample_**_xy
        """
        assert p3d.shape[-1] == 3
        p3d_flatten = p3d.reshape(-1, 3)
        if image_shape is None:
            image_shape = self.image_shape
        nviews = len(self.poses)
        p2d = np.zeros((nviews, p3d_flatten.shape[0], 2), dtype=np.float64) + np.nan   # nviews_nsample_2
        views = self.views
        if image_shape is not None:
            mask_inside = lambda proj: np.logical_and.reduce([proj[:,0]>0, proj[:,0]<image_shape[1],
                                                proj[:,1]>0, proj[:,1]<image_shape[0]])
        else:
            mask_inside = lambda proj: np.logical_and.reduce([proj[:,0]==proj[:,0], proj[:,1]==proj[:,1]])
        
        for view in views:
            param = lambda NAME : np.array(self.poses[view][NAME])
            K,R,t,dist = param('K'), param('R'), param('t'), param('dist')
            p2d_tmp, _ = project_points(p3d_flatten, K, R, t, dist, image_shape)
            p2d_tmp[~mask_inside(p2d_tmp)] = np.nan
            p2d[view] = p2d_tmp
        p2d = p2d.reshape((nviews, *p3d.shape[:-1], 2))
        return p2d

    def p2d_to_p3d(self, p2d:np.ndarray) -> np.ndarray:
        """
        Convert 2D points to 3D points.
        ---------- inputs ----------
        p2d: nview_nsample_**_xy
        image_shape: (H, W)

        ---------- outputs ----------
        p3d: nsample_**_xyz
        """
        original_shape = p2d.shape
        views = self.views
        nviews = len(views)
        assert original_shape[-1]==2
        assert nviews == original_shape[0]
        landmarks = p2d.reshape(original_shape[0], -1, original_shape[-1])
        p3d = build_input_np(views, self.poses, landmarks)
        p3d = p3d.reshape(*original_shape[1:-1], 3)
        return p3d


def build_input_np(views, poses:dict, landmarks:np.ndarray) -> np.ndarray:
    # transform camera poses to numpy array
    n_cameras = len(views)
    poses_list = [[np.squeeze(np.float64(poses[j][key])) for key in ['K', 'R', 't', 'dist']]
                    for j in range(n_cameras)]

    # undistort landmarks
    landmarks_undist_withnan = np.zeros_like(landmarks)
    for j in range(n_cameras):
        K,R,t,dist = poses_list[j]
        points = landmarks[views[j]]
        landmarks_undist_withnan[views[j]] = undistort_points(points, K, dist)

    # triangulate 3D positions from all possible pair of views
    p3d_allview_withnan = []
    for j1, j2 in itertools.combinations(range(n_cameras), 2):
        K1,R1,t1,dist1 = poses_list[j1]
        K2,R2,t2,dist2 = poses_list[j2]
        pts1 = landmarks_undist_withnan[views[j1]]
        pts2 = landmarks_undist_withnan[views[j2]]
        p3d = triangulate(pts1, pts2, K1, R1, t1, None, K2, R2, t2, None)
        p3d_allview_withnan.append(p3d)
    p3d_allview_withnan = np.array(p3d_allview_withnan) #nviewpairs_nsample_xyz

    # get median
    p3d = np.nanmedian(p3d_allview_withnan, axis=0) #nsample_xyz
    return p3d
