# from lilabnext.multiview_calib.p2d_raymatch ray_match_factory
#%%
import numpy as np
import cv2
import argparse
from lilab.multiview_scripts_dev.s6_calibpkl_predict import CalibPredict
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import linear_sum_assignment


def polor_ray(ba_poses, view_from, view_to):
    R0 = ba_poses[view_from]['R']
    t0 = ba_poses[view_from]['t'].reshape(3,1)
    K0 = ba_poses[view_from]['K']
    R1 = ba_poses[view_to]['R']
    t1 = ba_poses[view_to]['t'].reshape(3,1)
    K1 = ba_poses[view_to]['K']
    R0i = np.linalg.inv(R0)
    K0i = np.linalg.inv(K0)
    # s = [1, 0.1,2,5,10,50,100]
    s = [1.0, 100]
    K = K1@R1@R0i@K0i
    b = K1@t1 - K1@R1@R0i@t0
    def fun(uv1):
        uvd1 = np.array([uv1[0], uv1[1], 1])[:,None] * np.array(s)[None,:] #p2d[1]
        uvd2 = K@uvd1 + b   #等效于 uvd2 = K1@(R1@ (R0i@(K0i@uvd1 - t0)) + t1)
        uv2 = uvd2[:2]/uvd2[[2]]
        return uv2.T #nsamplex2
    return fun


def get_dist(get_ray_dict, p2d_src, p2d_dst, view_from, view_to, p2d_dst_real=None):
    get_ray = get_ray_dict[view_from, view_to]
    uv2_ray = np.array([get_ray(p2d) for p2d in p2d_src]) #nsrc, 端点2，xy
    n_src , n_dst = len(p2d_src), len(p2d_dst)
    assert n_src == n_dst
    A, B = uv2_ray[:,None,0,:], uv2_ray[:,None,1,:] #src 直线的两个端点
    T = p2d_dst[None]
    AB = B - A
    AT = T - A
    cost = np.cross(AB, AT, axis=-1)**2 / np.linalg.norm(AB+0.001, ord=2, axis=-1) #(n_src, n_dst)
    if np.all(np.isnan(cost)):
        cost[:] = 1.0
    else:
        cost[np.isnan(cost)] = np.nanmax(cost) * 10
    row_ind, col_ind = linear_sum_assignment(cost) #row_ind == np.arange(n_src)
    p2d_dst_real = p2d_dst if p2d_dst_real is None else p2d_dst_real
    p2d_dst_match = p2d_dst_real[col_ind]
    return uv2_ray, p2d_dst_match


def show(uv2_ray, p2d_dst_match):
    plt.figure()
    for i in range(len(p2d_dst_match)):
        plt.plot(uv2_ray[i,:,0], uv2_ray[i,:,1])
        plt.scatter(p2d_dst_match[i,0], p2d_dst_match[i,1])
    plt.xlim([0,640])
    plt.ylim([480,0])


def ray_match_factory(ba_poses:dict, nins:int):
    calibobj = CalibPredict({'ba_poses': ba_poses})
    nview = len(ba_poses)
    get_ray_dict = {(i_src,i_dst): polor_ray(ba_poses, i_src, i_dst) for i_src,i_dst
                in itertools.product(range(nview), range(nview))} #装饰器函数: view_from, view_to
    last_p3d = np.random.random((nins, 3)) + np.arange(nins)[:,None]

    def nodist_to_dist(xy:np.ndarray, iview:int) -> np.ndarray:
        params = calibobj.poses[iview]
        K, dist = params['K'], params['dist']
        uv = cv2.undistortPoints(xy.reshape(-1,2), K, dist, P=K)
        uv = uv.reshape(xy.shape)
        return uv
    
    def match_p3d(p3d:np.ndarray) -> np.ndarray:
        assert p3d.shape == (nins, 3)
        if np.isnan(p3d).any(): return last_p3d
        cost = np.linalg.norm(last_p3d[:,None] - p3d[None], axis=-1)  #(nins, nins)
        row_ind, col_ind = linear_sum_assignment(cost) #row_ind == np.arange(n_src)
        p3d_match = p3d[col_ind].astype(float) #(nins, 3)
        ind_isnan = np.isnan(p3d_match)
        if np.any(ind_isnan): p3d_match[ind_isnan] = last_p3d[ind_isnan]
        last_p3d[:] = p3d_match
        return last_p3d

    def p2d_ray_match_p3d(p2d:np.ndarray) -> np.ndarray:
        assert p2d.shape == (nview, nins, 2) #(nview, nins, xy)
        ind_valid_view = ~np.any(np.isnan(p2d[:,:,0]), axis=-1)
        if ind_valid_view.sum()<1:
            #return np.empty((nins,3), dtype=float)*np.nan
            p2d = calibobj.p3d_to_p2d(last_p3d)
        p2d_uv = np.array([nodist_to_dist(p2d[iview], iview) for iview in range(nview)])
        diff_2d = np.nan_to_num(np.linalg.norm(p2d[:,0,:] - p2d[:,-1,:], axis=-1)) #(nview,)
        view_from = np.argmax(diff_2d)
        p2d_match = np.array([get_dist(get_ray_dict, p2d_uv[view_from], p2d_uv[view_to], 
                                       view_from, view_to, p2d[view_to])[1]
                               for view_to in range(nview)], dtype=float)
        p3d_ray = calibobj.p2d_to_p3d(p2d_match) 
        p3d_match = match_p3d(p3d_ray) #(nins, 3)
        return p2d_match, p3d_match
    
    return p2d_ray_match_p3d
#%%
import pickle
import tqdm
import os.path as osp
nins = 2
#"\\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\marmoset_camera3_cxf\2024-4-10_Camera_calibration\Low\ball\ball_move.aligncalibpkl"
ball='/mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-4-10_Camera_calibration/Low/ball/ball_move.aligncalibpkl'
balldata=pickle.load(open(ball,'rb'))
matpkl='/mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-4-12-marmoset/2024-04-12_10-14-29_EdwardxLisa.matpkl'
matpkldata=pickle.load(open(matpkl,'rb'))
ba_poses = balldata['ba_poses']
convert = ray_match_factory(ba_poses, nins)
iframes=matpkldata['keypoints'].shape[1]

#%%
p3d = np.zeros((iframes, 2, 3))
p2d = np.zeros((6, iframes, 2, 2))
for i in tqdm.tqdm(range(iframes)):
    keypoints=matpkldata['keypoints'][:,i,:,:2]
    p3d[i, :, :] = convert(np.array(keypoints, dtype=np.float32))[1]
    if convert(np.array(keypoints, dtype=np.float32))[0].shape == (3,):
        p2d[:, i, :, :] = np.zeros((6, 2, 2))
    else:
        p2d[:, i, :, :] = convert(np.array(keypoints, dtype=np.float32))[0]

outdata = {
     'keypoints':matpkldata['keypoints'],
     'keypoints_xyz_ba': p3d,
     'keypoints_xy_ba': p2d,
     'views_xywh': matpkldata['views_xywh'],
     'info':matpkldata['info']
    }
showname= osp.splitext(osp.split(matpkl)[1])[0]+'_ray.matcalibpkl'
outpkl = osp.join(osp.dirname(matpkl), showname)
pickle.dump(outdata, open(outpkl, 'wb'))
# %%
