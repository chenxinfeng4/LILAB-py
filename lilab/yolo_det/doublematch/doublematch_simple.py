#%%
import pickle
import numpy as np
import ffmpegcv
import tqdm
import cv2
import os.path as osp
import argparse
from lilab.multiview_scripts_dev.s6_calibpkl_predict import CalibPredict
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import linear_sum_assignment

calibpkl='/mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-3-29/ballmove/nodist/ball_move.aligncalibpkl'
ball=pickle.load(open(calibpkl,'rb'))

matpkl='/mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-3-29/marmoset/twomarmoset.matpkl'
matdata=pickle.load(open(matpkl,'rb'))
ba_poses = ball['ba_poses']

#%%
# remove dist
for v in ba_poses.values(): v['dist'] *= 0
calibobj = CalibPredict({'ba_poses':ba_poses})


#%%
def polor_ray(view_from, view_to):
    R0 = ba_poses[view_from]['R']
    t0 = ba_poses[view_from]['t'].reshape(3,1)
    K0 = ba_poses[view_from]['K']
    R1 = ba_poses[view_to]['R']
    t1 = ba_poses[view_to]['t'].reshape(3,1)
    K1 = ba_poses[view_to]['K']
    R0i = np.linalg.inv(R0)
    K0i = np.linalg.inv(K0)
    s = [1, 0.1,2,5,10,50,100]
    K = K1@R1@R0i@K0i
    b = K1@t1 - K1@R1@R0i@t0
    def fun(uv1):
        uvd1 = np.array([uv1[0], uv1[1], 1])[:,None] * np.array(s)[None,:] #p2d[1]
        uvd2 = K@uvd1 + b   #等效于 uvd2 = K1@(R1@ (R0i@(K0i@uvd1 - t0)) + t1)
        uv2 = uvd2[:2]/uvd2[[2]]
        return uv2.T #nsamplex2
    return fun


nview = len(ba_poses)
get_ray_dict = {(i_src,i_dst): polor_ray(i_src, i_dst) for i_src,i_dst
                in itertools.product(range(nview), range(nview))} #装饰器函数: view_from, view_to

#%%
def get_dist(p2d_src, p2d_dst, view_from, view_to):
    get_ray = get_ray_dict[view_from, view_to]
    uv2_ray = np.array([get_ray(p2d) for p2d in p2d_src]) #nsrc, 端点2，xy
    n_src , n_dst = len(p2d_src), len(p2d_dst)
    assert n_src == n_dst
    A, B = uv2_ray[:,None,0,:], uv2_ray[:,None,1,:] #src 直线的两个端点
    T = p2d_dst[None]
    AB = B - A
    AT = T - A
    cost = np.cross(AB, AT, axis=-1)**2 / np.linalg.norm(AB, ord=2, axis=-1) #(n_src, n_dst)
    cost[np.isnan(cost)] = np.nanmax(cost) * 10
    row_ind, col_ind = linear_sum_assignment(cost) #row_ind == np.arange(n_src)
    p2d_dst_match = p2d_dst[col_ind]
    return uv2_ray, p2d_dst_match


def show(uv2_ray, p2d_dst_match):
    plt.figure()
    for i in range(len(p2d_dst_match)):
        plt.plot(uv2_ray[i,:,0], uv2_ray[i,:,1])
        plt.scatter(p2d_dst_match[i,0], p2d_dst_match[i,1])
    plt.xlim([0,640])
    plt.ylim([480,0])


# p2d_nview_raw = calibobj.p3d_to_p2d(ball['keypoints_xyz_ba'][141])
# p2d_src = p2d_nview_raw[view_from]
# p2d_dst = p2d_nview_raw[view_to]
# uv2_ray, p2d_dst_match = get_dist(p2d_src, p2d_dst)
# show(uv2_ray, p2d_dst_match)

#%%
view_from = 2
view_to =3
keypoints_xy = matdata['keypoints'][...,:2]
p2d_src_nt = keypoints_xy[view_from]
p2d_dst_nt = keypoints_xy[view_to]
p2d_dst_nt_shuffle = p2d_dst_nt[:,[1,0],:]

p2d_dst_nt_match = np.array([get_dist(p2d_src, p2d_dst, view_from, view_to)[1]
                    for p2d_src, p2d_dst in zip(p2d_src_nt, p2d_dst_nt_shuffle)])

assert np.all(np.isclose(p2d_dst_nt_match, p2d_dst_nt, equal_nan=True))

#%%

#%%
view_from = 1
view_to = 3
keypoints_xy = matdata['keypoints'][...,:2]
p2d_src_nt = keypoints_xy[view_from]
p2d_dst_nt = keypoints_xy[view_to]
p2d_src, p2d_dst = p2d_src_nt[2500], p2d_dst_nt[2500]
uv2_ray, p2d_dst_match = get_dist(p2d_src, p2d_dst, view_from, view_to)
show(uv2_ray, p2d_dst_match)

iframe = 2500
nanimal = keypoints_xy.shape[2]
p2d_nviews = keypoints_xy[:, iframe]      #nview, nanimal, xy
p2d_ndet = np.sum(~np.isnan(p2d_nviews[...,0]), axis=-1)
isvalid_slice = np.sum(p2d_ndet==nanimal)>=2
ind_view_from = np.nanargmax(np.linalg.norm(p2d_nviews[:,0]-p2d_nviews[:,1], axis=-1))

p2d_src = p2d_nviews[ind_view_from]
p2d_dst_nt_match = np.zeros((nview, nanimal, 2))
for view_to in range(nview):
    p2d_dst = p2d_nviews[view_to]
    _, p2d_dst_nt_match[view_to] = get_dist(p2d_src, p2d_dst, ind_view_from, view_to)

for view_to in range(nview):
    A, B = p2d_dst_nt_match[view_to]
    plt.figure()
    plt.scatter(A[0], A[1])
    plt.scatter(B[0], B[1])
    plt.xlim([0,640])
    plt.ylim([480,0])

p3d = calibobj.p2d_to_p3d(p2d_dst_nt_match)
p2d_reproj = calibobj.p3d_to_p2d(p3d)
for view_to in range(nview):
    A, B = p2d_reproj[view_to]
    plt.figure()
    plt.scatter(A[0], A[1])
    plt.scatter(B[0], B[1])
    plt.xlim([-100,640])
    plt.ylim([480,-100])

#%%
import mmcv
vfile='/mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-3-29/marmoset/twomarmoset.mp4'
vid = mmcv.VideoReader(vfile)
frame = vid[iframe]
views_xy = np.array(matdata['views_xywh'])[:,:2]

p2d_canvas = p2d_reproj + views_xy[:,None,:]
plt.figure(figsize=(20,14))
plt.imshow(frame[...,::-1])
plt.scatter(p2d_canvas[:,0,0], p2d_canvas[:,0,1], color='r')
plt.scatter(p2d_canvas[:,1,0], p2d_canvas[:,1,1], color='C0')
plt.axis('off')