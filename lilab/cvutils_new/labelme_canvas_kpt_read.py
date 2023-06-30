# %%
import os.path as osp
import argparse
from glob import glob 
import cv2
from lilab.cameras_setup import get_view_xywh_1280x800x10 as get_view_xywh
import json
import numpy as np
import pickle
from collections import defaultdict
from scipy.optimize import minimize

file = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wxw_test/label/2022-05-15_18-41-53_wwwbrat_000240.json'
img  = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wxw_test/label/2022-05-15_18-41-53_wwwbrat_000240.jpg'
calibfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/OXTRHETxKO/2022-05-11ball.calibpkl'

# %%
data = json.load(open(file, 'r'))
calibdata = pickle.load(open(calibfile, 'rb'))
# %%
in_data = dict()
for shape in data['shapes']:
    label = shape['label']
    point = shape['points'][0]
    in_data[label] = in_data.get(label, [])
    in_data[label].append(point)


views = get_view_xywh()

n_view = len(views)
n_animal = 4
n_keytype = len(in_data)
outdata = np.ones((n_view, n_keytype, n_animal, 2)) + np.nan


def locate_keypoint(views, point):
    for i, view in enumerate(views):
        x, y, w, h = view
        if x<=point[0]<=x+w and y<=point[1]<=y+h:
            return i, [point[0]-x, point[1]-y]
    else:
        raise ValueError('point not in any view')

keytypes = ['Nose', ]
inddict = defaultdict(lambda: -1)
for ikey, keytype in enumerate(keytypes):
    points = in_data[keytype]
    for point in points:
        iview, kpt = locate_keypoint(views, point)
        inddict[(iview, ikey)] += 1
        outdata[iview][ikey][inddict[(iview, ikey)]] = kpt



# %%
ba_global_params = calibdata['ba_global_params']
kargs = calibdata['ba_global_params']['kargs']
fun_global2ba = lambda p3d_global: eval(ba_global_params['strfun_global2ba'])(p3d_global, kargs)
fun_ba2global = lambda p3d_ba: eval(ba_global_params['strfun_ba2global'])(p3d_ba, kargs)

# %%
seeds_global = np.array([[0,0,2], 
                         [0,10, 2], 
                         [10,0, 2], 
                         [10,10, 2],
                         [0, 20, 2],
                         [10, 20, 2],
                         [20, 10, 2],
                         [20, 0, 2],
                         [20, 20, 2],])
seeds_ba = fun_global2ba(seeds_global)
seed_ba = seeds_ba[0]

# %%
ba_poses = calibdata['ba_poses']
for value1 in ba_poses.values():
    for key2 in value1.keys():
        value1[key2] = np.array(value1[key2])

# %%
nose = outdata[:, 0, :] #nview_nobj_2

from multiview_calib.singleview_geometry import project_points


# %%
def project_p3d_to_p2d(views, poses, p3d):
    nviews = len(poses)
    p2d = np.empty((nviews, p3d.shape[0], 2)) + np.nan   # nviews_nsample_2

    for view in views:
        param = lambda NAME : poses[view][NAME]
        K,R,t,dist = param('K'), param('R'), param('t'), param('dist')
        p2d[view], _ = project_points(p3d, K, R, t, dist)
    return p2d

def loss_func(p2d, p2d_anno):
    # p2d_anno: nviews_nobj_2
    p2d_single = p2d[:,0,:] # nviews_2
    p2d_single = p2d_single[:, np.newaxis, :] # nviews_1_2
    distance = p2d_anno - p2d_single # nviews_nobj_2
    distance_norm = np.linalg.norm(distance, axis=2) # nviews_nobj
    distance_min = np.nanmin(distance_norm, axis=1) # nviews
    loss = np.nanmean(distance_min)
    print(loss)
    if np.isnan(loss):
        loss = 0
    return loss
    
def cal_loss(p3d, views, poses, p2d_anno):
    p3d_ba = fun_global2ba(p3d[None,:])
    p2d = project_p3d_to_p2d(views, poses, p3d_ba)
    loss = loss_func(p2d, p2d_anno)
    return loss

# %%
views_num = range(10)
p2d_anno =nose
poses = ba_poses
res = minimize(cal_loss, x0=seeds_global[0], 
                    args=(views_num, poses, p2d_anno),
                    options={'maxiter':100})

def project_globalp3d_to_p2d(views, poses, p3d):
    p3d = fun_global2ba(p3d[None,:])
    nviews = len(poses)
    p2d = np.empty((nviews, p3d.shape[0], 2)) + np.nan   # nviews_nsample_2

    for view in views:
        param = lambda NAME : poses[view][NAME]
        K,R,t,dist = param('K'), param('R'), param('t'), param('dist')
        p2d[view], _ = project_points(p3d, K, R, t, dist)
    return p2d


def plot(img, p2d, views):
    img_np = cv2.imread(img)
    for point, crop_xy in zip(p2d, views):
        point = point.astype(int)[0]
        x, y, w, h = crop_xy
        point_canvas = point + np.array([x, y])
        cv2.circle(img_np, tuple(point_canvas), 5, (0,0,255), -1)

    outfile = osp.splitext(img)[0]+'_out.png'
    cv2.imwrite(outfile, img_np)


p3d = res['x']
p2d = project_globalp3d_to_p2d(views_num, poses, p3d)
plot(img, p2d, views)


# %%

def nms_filter(p3d, p2d_anno, nms_thr):
    p2d = project_globalp3d_to_p2d(views_num, poses, p3d) # nviews_1_2
    p2d_single = p2d      # nviews_1_2
    distance = p2d_anno - p2d_single # nviews_nobj_2
    distance_norm = np.linalg.norm(distance, axis=2) # nviews_nobj
    distance_norm[np.isnan(distance_norm)] = nms_thr # nviews_nobj
    distance_argmin = np.expand_dims(np.argmin(distance_norm, axis=1), axis=1) # nviews_1
    distance_min = np.take_along_axis(distance_norm, distance_argmin, axis=1)[:,0] # nviews
    under_thr = distance_min <= nms_thr            # nviews
    p2d_anno = p2d_anno.copy()
    for i in range(len(views_num)):
        if under_thr[i]:
            p2d_anno[i][distance_argmin[i][0]] = np.nan
    return p2d_anno

p2d_anno_next = nms_filter(p3d, p2d_anno, nms_thr=10)

res = minimize(cal_loss, x0=seeds_global[0], 
                    args=(views_num, poses, p2d_anno_next),
                    options={'maxiter':100})

p3d_next = res['x']
p2d_project = project_globalp3d_to_p2d(views_num, poses, p3d_next)
# %%
colors = ( (0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0))
def plotobjs(img, views, p2d_merge):
    img_np = cv2.imread(img)
    for color, p2d in zip(colors, p2d_merge):
        for point, crop_xy in zip(p2d, views):
            point = point.astype(int)[0]
            x, y, w, h = crop_xy
            point_canvas = point + np.array([x, y])
            cv2.circle(img_np, tuple(point_canvas), 5, color, -1)

    outfile = osp.splitext(img)[0]+'_out2.jpg'
    cv2.imwrite(outfile, img_np)
plotobjs(img, views, [p2d, p2d_project])



# %%
p2d_anno_last = p2d_anno
p3d_merge = []
for i in range(n_animal):
    res = minimize(cal_loss, x0=seeds_global[0], 
                        args=(views_num, poses, p2d_anno_last),
                        options={'maxiter':100})
    p3d = res['x']
    p2d_anno_last = nms_filter(p3d, p2d_anno_last, nms_thr=10)
    p3d_merge.append(p3d)

p2d_merge = [project_globalp3d_to_p2d(views_num, poses, p3d) for p3d in p3d_merge]


# %%
def plotobjs(img, views, p2d_merge):
    img_np = cv2.imread(img)
    for color, p2d in zip(colors, p2d_merge):
        for point, crop_xy in zip(p2d, views):
            point = point.astype(int)[0]
            x, y, w, h = crop_xy
            point_canvas = point + np.array([x, y])
            cv2.circle(img_np, tuple(point_canvas), 10, color, -1)

    outfile = osp.splitext(img)[0]+'_out2.jpg'
    cv2.imwrite(outfile, img_np)
plotobjs(img, views, p2d_merge)

# %%
import copy
project_p3d_to_p2d([0], poses, seed_ba[None,:])

poses2 = copy.deepcopy(poses)

seed0 = seeds_global[0]

# R1 = np.array([0.1,1.45, 2, 0.2, 1.3, 1.3, 1.3, 4, 1.6], dtype=float).reshape(3,3)
R1 = np.array([2, 0, 0, 0, 2, 0, 0, 0, 2], dtype=float).reshape(3,3)
t1 = np.array([0, 0, 0]).reshape(3,1)
for key in poses.keys():
    t = poses[key]['t'].reshape(3,1)
    R = poses[key]['R']
    # R1 =  kargs['R'].T
    # t1 = -kargs['t'].reshape(3,1)
    R2 = R @ R1
    t2 = R @ t1 + t
    poses2[key]['R'] = R2
    poses2[key]['t'] = t2.reshape(3,)

R = poses[0]['R']
t = poses[0]['t']
K = poses[0]['K']
dist = poses[0]['dist']*0


R_newd = np.array([[2, 0, 0], 
                    [0, 2, 0], 
                    [0, 0, 2]], dtype=float).reshape(3,3)
t_newd = np.array([1, 11, 17]).reshape(3,1)
p3d_new = np.array([10,10,-30]).reshape(3,1)

p3d_old = R_newd @ p3d_new + t_newd  # (3,1)


# 2d
pts_ = p3d_old.reshape(-1,3)
np.dot(R, pts_.T) + t.reshape(3,-1)

# 2d
pts_ = p3d_new.reshape(-1,3)
R2 = R @ R_newd
t2 = R @ t_newd + t.reshape(3,-1)
np.dot(R2, pts_.T) + t2.reshape(3,-1)

# %%
project_points(p3d_new.T.astype(float), K, R2, t2, dist)
project_points(p3d_old.T.astype(float), K, R, t, dist)
# %%
