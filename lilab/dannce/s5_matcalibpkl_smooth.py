# %%
import pickle
from scipy.signal import medfilt
from scipy.ndimage import convolve1d
import numpy as np
import os.path as osp
import argparse
mat_file = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/2021-11-02-bwrat_800x600/30fps/2021-11-02_15-29-56.matcalibpkl'

def convert(mat_file):
    data = pickle.load(open(mat_file, 'rb'))

    # %%
    data_seq = data['keypoints_xyz_ba']
    # interpolate the missing frames

    keypoints_xyz_ba = convolve1d(data_seq, np.ones(5)/5, 
                                        axis=0, mode='nearest')

    data_seq = data['keypoints_xy_ba']
    keypoints_xy_ba = convolve1d(data_seq, np.ones(5)/5, 
                                        axis=1, mode='nearest')

    data['keypoints_xyz_ba'] = keypoints_xyz_ba
    data['keypoints_xy_ba'] = keypoints_xy_ba

    outpickle = osp.splitext(mat_file)[0]+'_smooth.matcalibpkl'
    pickle.dump(data, open(outpickle, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mat_file', type=str)
    args = parser.parse_args()
    mat_file = args.mat_file
    assert osp.exists(mat_file), 'mat_file not exists'
    convert(mat_file)