# ls *matcalibpkl | xargs -n 1 python -m lilab.dannce.s5_smoothmat2matcalibpkl
# %%
import os.path as osp
import pickle
import argparse
import numpy as np
import scipy.io as sio


# %%
matcalib = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/2021-11-02-bwrat_800x600/30fps/2021-11-02_15-11-45.matcalibpkl'

classnames = ['black', 'white']

def load_postfix_mat(matcalib, postfix):
    files = [osp.splitext(matcalib)[0] + f'_{classname}_{postfix}.mat' for classname in classnames]
    assert all([osp.exists(f) for f in files]), 'Not all files exist'
    data_3d = np.array([sio.loadmat(f)['points_3d'] for f in files])
    data_3d = data_3d.transpose(1, 0, 2, 3)
    return data_3d

def convert(matcalib):
    data = pickle.load(open(matcalib, 'rb'))
    keypoints_xyz_ba_1impute = load_postfix_mat(matcalib, '1impute')
    keypoints_xyz_ba_2outlierfree = load_postfix_mat(matcalib, '2outlierfree')
    keypoints_xyz_ba_3smooth = load_postfix_mat(matcalib, '3smooth')
    data['keypoints_xyz_ba_1impute'] = keypoints_xyz_ba_1impute
    data['keypoints_xyz_ba_2outlierfree'] = keypoints_xyz_ba_2outlierfree
    data['keypoints_xyz_ba_3smooth'] = keypoints_xyz_ba_3smooth
    outpkl = osp.splitext(matcalib)[0] + '_smooth.matcalibpkl'
    pickle.dump(data, open(outpkl, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('matcalib', type=str)
    args = parser.parse_args()
    matcalib = args.matcalib
    assert osp.exists(matcalib), 'matcalib not exists'
    convert(matcalib)
