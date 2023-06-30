# python -m lilab.dannce.s1_matcalibpkl2anno
# %%
import os.path as osp
import pickle
import numpy as np
import scipy.io as sio
import argparse

dir = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/2021-11-02-bwrat_800x600/30fps/outframes/'
matcalibpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/TPH2-KO-multiview-202201/male/ball/ball.matcalibpkl'

# %%

from lilab.cvutils_new.parse_name_of_extract_frames import parsenames_indir


def pack_anno_data(imageNames, data_3D, out_anno_mat=None):
    assert len(imageNames) == data_3D.shape[0]
    if data_3D.ndim == 3:
        data_3D = data_3D.reshape((data_3D.shape[0], -1))
    assert data_3D.ndim == 2
    assert data_3D.shape[1] % 3 == 0
    imageNames = np.array(imageNames, dtype=object)
    dataout = {'imageNames': imageNames, 'data_3D': data_3D}

    if out_anno_mat is not None:
        sio.savemat(out_anno_mat, dataout)
    return dataout


def pack_calib_data(ba_poses, out_calibpkl_mat=None):
    ba_poses_ = [ba_poses[i] for i in range(len(ba_poses))]
    dataout = {'ba_poses': ba_poses_}

    if out_calibpkl_mat is not None:
        sio.savemat(out_calibpkl_mat, dataout)
    return dataout


def convert(dir):
    imagebasenames, parsedinfo = parsenames_indir(dir)

    # %%
    assert len(set([info[0] for info in parsedinfo]))==1
    videoname = parsedinfo[0][0]
    videonakename = osp.splitext(osp.basename(videoname))[0]
    assert videonakename == osp.splitext(osp.basename(matcalibpkl))[0]

    # %%
    outcalibpkl_mat = osp.join(dir, videonakename+'.calibpkl.mat')

    ba_poses = pickle.load(open(matcalibpkl, 'rb'))['ba_poses']
    pack_calib_data(ba_poses, outcalibpkl_mat)


    # %%
    anno_mat = osp.join(dir, 'anno_matcalibpkl.mat')
    keypoints_xyz_ba = data['keypoints_xyz_ba']
    pts3d = [keypoints_xyz_ba[info[2]//3, info[1]] for info in parsedinfo]
    pts3d = np.array(pts3d) #(N, K, 3)
    data_3D = np.ascontiguousarray(pts3d.reshape((pts3d.shape[0], -1)))
    imageNames = np.array(imagebasenames, dtype=object)
    dataout = {'imageNames': imageNames, 'data_3D': data_3D}

    sio.savemat(anno_mat, dataout)


# %%
