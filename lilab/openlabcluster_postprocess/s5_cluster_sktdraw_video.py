
# conda activate mmdet
# %%
import os
import scipy.io as sio
import pickle
import tqdm
import numpy as np
import os.path as osp
import glob
from collections import defaultdict
import random
import multiprocessing as mp
import re
import matplotlib.pyplot as plt
from lilab.mmpose_dev.a3_ego_align import KptEgoAligner
from matplotlib.patches import Polygon
import argparse
from lilab.openlabcluster_postprocess.s4_moseq_like_motif_plot import parsename

from lilab.multiview_scripts.ratpoint3d_to_video import *
from lilab.mmpose.s3_matcalibpkl_2_video import plot_video

#%%
##get origin video data
fps=30
time_long=15*60 #seconds

matcalib_dir = '/DATA/taoxianming/rat/data/chenxinfeng/sexDay55/smoothfoot_all'
clippredpklfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/sexDay55/kmeans/FWPCA0.00_P100_en3_hid30_epoch777_svm2allAcc0.93_kmeansK2use-38_fromK1-20_K100.clippredpkl'


def main(clippredpklfile, matcalib_dir):
    clipdata = pickle.load(open(clippredpklfile, 'rb'))
    assert {'ncluster', 'ntwin', 'cluster_labels', 'embedding', 'embedding_d2', 'clipNames'}

    #%%
    clipdata['clipNames']
    parsename(clipdata['clipNames'][0])
    isblack = np.array(['blackFirst' in f for f in clipdata['clipNames']])
    clipName = clipdata['clipNames'][isblack]
    cluster_labels = clipdata['cluster_labels'][isblack]
    assert cluster_labels.min()==1   # start from 1. The 0 is non social
    nsample_eachcluster = 100        # 100clips

    ## 随机在每类别中挑选 100个片段
    clipName_downsample = []
    NCluster = clipdata['ncluster']
    for i in np.arange(NCluster)+1:
        ind_cluster = np.where(cluster_labels == i)[0]
        replace = True if len(ind_cluster) < nsample_eachcluster else False
        ind_downsample_cluster = np.random.choice(ind_cluster, nsample_eachcluster, replace=replace)
        clipName_downsample.append(clipName[ind_downsample_cluster])

    clip_video_cluster_l = []  # list of [iCluster, vnakename, iframe]
    for iCluster, clipName_downsample_this in enumerate(clipName_downsample, start=1):
        for j, clipName_this in enumerate(clipName_downsample_this):
            vnakename, iframe = parsename(clipName_this)
            clip_video_cluster_l.append([iCluster, vnakename, iframe])


    ## 找到每个片段对应的3D关键点坐标
    vnakename_l = list(set(c[1] for c in clip_video_cluster_l))
    vname_l = [osp.join(matcalib_dir, f+'.smoothed_foot.matcalibpkl') for f in vnakename_l]
    assert all(osp.isfile(f) for f in vname_l)
    keypoints_xyz_l = [pickle.load(open(f, 'rb'))['keypoints_xyz_ba'] for f in vname_l]
    keypoints_xyz_dict = dict(zip(vnakename_l, keypoints_xyz_l))

    outvideo_dir = osp.join(osp.dirname(clippredpklfile), 'cluster_sktdraw')
    os.makedirs(outvideo_dir, exist_ok=True)
    outvideo_file_list = [osp.join(outvideo_dir, f'Cluster_{i+1}.mp4') for i in range(NCluster)]

    pts3d = [[] for _ in range(NCluster)]
    for iCluster, vname, iframe in clip_video_cluster_l: #iCluster start from 1
        pts3d_this = keypoints_xyz_dict[vname][iframe:iframe+24] #seglen_nanimal_14_3
        pts3d[iCluster-1].append(pts3d_this)

    pts3d_np = np.array(pts3d)  # NCluster_ndownsample_seglen_nanimal_14_3
    assert pts3d_np.shape == (NCluster, nsample_eachcluster, pts3d_np.shape[2], 2, 14, 3)

    pts3d_seq = np.reshape(pts3d_np, (pts3d_np.shape[0], pts3d_np.shape[1]*pts3d_np.shape[2],
                                    *pts3d_np.shape[3:]))  # NCluster_mseglen_nanimal_14_3

    ## 开始画图。利用多进程可以并行加速
    pool = mp.Pool(processes=10)

    for outfile, pts3d_seq_this in zip(outvideo_file_list, pts3d_seq):
        pts3d_black, pts3d_white = pts3d_seq_this[:,0], pts3d_seq_this[:,1]
        pool.apply_async(plot_video, args=(pts3d_black, pts3d_white, outfile, fps))

    # 关闭进程池，不再接受新任务
    pool.close()
    # 等待所有子进程完成
    pool.join()
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("clippredpkl", type=str)
    parser.add_argument("matcalib_dir", type=str, help='Folder of all the *smoothfoot.matcalib')
    args = parser.parse_args()
    assert osp.isfile(args.clippredpkl)
    assert osp.isdir(args.matcalib_dir)
    main(args.clippredpkl, args.matcalib_dir)
