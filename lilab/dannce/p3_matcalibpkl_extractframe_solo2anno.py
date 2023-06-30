# python -m lilab.dannce.p3_matcalibpkl_extractframe_solo2anno IMAGEDIR MATCALIBDIR
# %%
import os.path as osp
import pickle
import numpy as np
import scipy.io as sio
import argparse
import glob
import pandas as pd
from lilab.cvutils_new.parse_name_of_extract_frames import parsenames_indir
import argparse

imagedir = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY35SOLO/outframes'
matcalibdir = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY35SOLO/'

# %%
matdata_dict = dict()
pkldata = None

def quary_pt3d(video_name, frame_id, matcalibnakenames, ratid_dict):
    if video_name not in matdata_dict:
        global pkldata
        pkldata_now = pickle.load(open(matcalibnakenames[video_name], 'rb'))
        pkldata = pkldata_now
        kpt_allrat = pkldata['keypoints_xyz_ba']
        Nframe, Nrat, Nkpt, Ndim = kpt_allrat.shape
        assert Nrat==2 and Nframe>100 and Ndim==3
        kpt_rat = kpt_allrat[:,ratid_dict[video_name],...]
        kpt_rat = kpt_rat.reshape(Nframe, -1)
        matdata_dict[video_name] = kpt_rat

    kpt_rat = matdata_dict[video_name]
    kpt_iframe = kpt_rat[frame_id]
    return kpt_iframe


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


def convert(dir, matcalibdir, force_ratid):
    # load data
    matcalibpkls = glob.glob(osp.join(matcalibdir, '*.matcalibpkl'))
    assert matcalibpkls

    imagebasenames, parsedinfo = parsenames_indir(dir)
    df = pd.DataFrame(parsedinfo, columns = ['video_name', 'rat_id', 'frame_id'])
    matcalibnakenames = {osp.basename(f).split('.')[0]+'.mp4':f for f in matcalibpkls}

    assert len(df)
    assert set(df['video_name'].to_list()) <= matcalibnakenames.keys()
    if force_ratid:
        assert force_ratid in ['w', 'b']
        ratid_dict = {f: int(force_ratid=='w') for f in matcalibnakenames.keys()}
    else:
        assert all(f[20] in ['w', 'b'] for f in matcalibnakenames.keys())
        ratid_dict = {f: int(f[20]=='w') for f in matcalibnakenames.keys()}

    # get data_3D
    data_3D = [quary_pt3d(v, f, matcalibnakenames, ratid_dict) 
               for v, f in zip(df['video_name'], df['frame_id'])]
    data_3D = np.array(data_3D, dtype=np.float64)
    assert data_3D.ndim == 2

    # save data to mat
    outannofile = osp.join(dir, 'anno_gene.mat')
    out_calibpkl_mat = osp.join(dir, 'ball.calibpkl.mat')
    print(outannofile)
    print(out_calibpkl_mat)
    pack_anno_data(imagebasenames, data_3D, outannofile)
    pack_calib_data(pkldata['ba_poses'], out_calibpkl_mat=out_calibpkl_mat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('imagedir', type=str)
    parser.add_argument('matcalibdir', type=str)
    parser.add_argument('--force-ratid', type=str, default=None)
    args = parser.parse_args()
    assert osp.isdir(args.imagedir)
    assert osp.isdir(args.matcalibdir)
    assert args.force_ratid in [None, 'w', 'b']
    convert(args.imagedir, args.matcalibdir, args.force_ratid)
