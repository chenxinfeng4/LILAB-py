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
from multiview_calib.bundle_adjustment_scipy import project_points
from lilab.multiview_scripts_dev.s4_matpkl2matcalibpkl import build_input_short


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
        assert {'ba_poses'} < set(calib.keys()), 'calib file is not correct'
        self.poses = calib['ba_poses']
        self.image_shape = list(calib['intrinsics'].values())[0]['image_shape'] if 'intrinsics' in calib else None #HxW
        self.views = sorted(list(calib['ba_poses'].keys()))

    def p3d_to_p2d(self, p3d, image_shape=None):
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
        p2d = np.empty((nviews, p3d_flatten.shape[0], 2)) + np.nan   # nviews_nsample_2
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

    def p2d_to_p3d(self, p2d):
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
        landmarks = {i: landmarks[i] for i in range(original_shape[0])}
        p3d = build_input_short(views, self.poses, landmarks)
        p3d = p3d.reshape(*original_shape[1:-1], 3)
        return p3d
    