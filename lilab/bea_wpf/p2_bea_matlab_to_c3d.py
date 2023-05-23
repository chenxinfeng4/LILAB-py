#%%                        
import numpy as np
import pickle
import os.path as osp
import pandas as pd
import scipy.io as sio
import argparse
from lilab.bea_wpf.a2_matcalibpkl_to_c3d import convert_kpt_3d_to_c3d

matfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zzc_to_dzy/openlabcluster-result20230330/11.mat'
sampleinfopkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zzc_to_dzy/20230222_oxtr-gq_m/multi/sub_2video/2023-02-23_16-57-04_a1p2_merge_samplesinfo.pkl'

# %%
def main(matfile, sampleinfopkl):
    matdata = sio.loadmat(matfile)
    label_seq = np.squeeze(matdata['label_seq'])
    # %%
    # 'video_nake_name', 'frameid', 'label'
    datapkl = pickle.load(open(sampleinfopkl, 'rb'))['nsample_l']

    kpt_3d_orig_l = []
    for matcalibpkl, nframe in datapkl:
        data = pickle.load(open(matcalibpkl, 'rb'))
        kpt_3d_orig = data['keypoints_xyz_ba']
        assert nframe ==len(kpt_3d_orig), 'Not equal length'
        kpt_3d_orig_l.append(kpt_3d_orig)

    kpt_3d_orig = np.concatenate(kpt_3d_orig_l, axis=0)
    assert len(kpt_3d_orig) == label_seq.shape[0], 'Not equal length'

    outc3dfile = osp.splitext(sampleinfopkl)[0] + '.clip_framebyframe_labels.c3d'
    convert_kpt_3d_to_c3d(kpt_3d_orig, outc3dfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('matfile', type=str)
    parser.add_argument('sampleinfopkl', type=str)
    args = parser.parse_args()
    main(args.matfile, args.sampleinfopkl)
