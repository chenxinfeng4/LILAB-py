import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.stats
import scipy.io as sio
from  scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from scipy.stats import norm

from lilab.multiview_scripts.ratpoint3d_to_video import *
from ipywidgets import interact, interactive, fixed, interact_manual
from lilab.smoothpoints.LSTM_point3d_impute_pred import predict
from scipy.signal import medfilt
from scipy.ndimage import convolve1d

mat_file = '/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_side6/ratBblack/rat_points3d_cm_impute.mat'

# mat_file = '/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_side6/rat/rat_points3d_cm_impute.mat'

def plot_skeleton_aframe_easy(point3d_aframe):
    fig, ax = init_3d_plot()
    plot_skeleton_aframe(None, name='white', createdumpy=True)
    plot_skeleton_aframe(point3d_aframe, name='white', createdumpy=False)
    return fig, ax

def plot_skeleton_aframe_black(point3d_aframe):
    plot_skeleton_aframe(None, name='black', createdumpy=True)
    plot_skeleton_aframe(point3d_aframe, name='black', createdumpy=False)

def index_to_bodypart(index):
    bodys = ['Nose', 'EarL', 'EarR', 'Neck','Back', 'Tail', 'FShoulderL', 'FPawL', 
            'FShoulderR', 'FPawR', 'BShoulderL', 'BPawL', 'BShoulderR', 'BPawR']
    strlist = []
    for i, body in zip(index, bodys):
        if i == 1:
            strlist.append(body)
        elif i>1:
            strlist.append(f'{i}x{body}')
        else:
            pass
    return ' '.join(strlist)

def kickout_and_hybrid(mat_data):
    mat_dist = np.array([pdist(skt) for skt in mat_data]) #1802x91,  91=all pairs distance
    mat_3std = np.zeros_like(mat_dist)  # 1802x91, [0 | 1], outliers for 3 std
    for dist, mat_3std_current in zip(mat_dist.T, mat_3std.T):
        distribute = norm.fit(dist)
        thrs = distribute[0] + np.array([-1, 1])*3*distribute[1]
        outliers = np.logical_or(dist<thrs[0], dist>thrs[1])
        mat_3std_current[outliers] = 1
        
    mat_3std_fullatrix = np.array([squareform(M)
                                    for M in mat_3std]) # 1802x14x14, 3-std outliers

    mat_3std_count = np.sum(mat_3std_fullatrix, axis = 1)  # 1802x14, sumery the 3-std outliers of all pairs for each body part

    mat_data_kickout = mat_data.copy()
    max_n = 1
    thr_count = 2
    max_n_outliermask = np.zeros(mat_data.shape[1], dtype=bool)
    max_n_outliermask[:max_n] = True

    for frame_outliers, skt_kickout in zip(mat_3std_count, mat_data_kickout):
        sortind = np.argsort(frame_outliers)[::-1] # descending order
        frame_outliers_sort = frame_outliers[sortind]
        outlier = frame_outliers_sort >= thr_count
        outlier_max_n = np.logical_and(outlier, max_n_outliermask)
        ind_outlier = sortind[outlier_max_n]
        skt_kickout[ind_outlier,:] = np.nan

    n_kickout = np.sum(np.isnan(mat_data_kickout[:,:,0]))
    print('Kickout {} points'.format(n_kickout))

    # %% get the imputation
    mat_data_impute, mat_data_hybrid = predict(mat_data_kickout)

    return mat_data_hybrid


def main(mat_file):
    mat_data = sio.loadmat(mat_file)['points_3d']
    assert np.all(~np.isnan(mat_data)), 'NaN in mat_data.'
    mat_data_hybrid = kickout_and_hybrid(mat_data)
    mat_data_hybrid = kickout_and_hybrid(mat_data_hybrid)

    outfile = osp.join(osp.dirname(mat_file), 'rat_points3d_cm_2outlierfree.mat')
    sio.savemat(outfile, {'points_3d': mat_data_hybrid})

    # %% filter to smooth
    smooth_method = 'mean'
    if smooth_method == 'median':
        mat_data_smooth = medfilt(mat_data_hybrid, kernel_size=(5,1,1))
    elif smooth_method == 'mean':
        mat_data_smooth = convolve1d(mat_data_hybrid, np.ones(3)/3, 
                                     axis=0, mode='nearest')
    outfile = osp.join(osp.dirname(mat_file), 'rat_points3d_cm_3smooth.mat')
    sio.savemat(outfile, {'points_3d': mat_data_smooth})


if __name__ == '__main__':
    main(mat_file)
    