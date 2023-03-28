# conda activate mmpose
# python -m lilab.smoothnet.s1_matcalibpkl2smooth_foot xxx.matcalibpkl
# %% imports
import pickle
from mmpose.core.post_processing.temporal_filters import build_filter
from lilab.multiview_scripts_dev.s6_calibpkl_predict import CalibPredict
import numpy as np
import argparse

# %%
pklfile='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-26_19-31-21_SHANK20_HetxHet.matcalibpkl'

checkpoint='/home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/smoothnet/checkpoint/smoothnet_ws64_cxf.pth'
checkpoint_foot='/home/liying_lab/chenxinfeng/DATA/SmoothNet/results/h36m_cxf_16b3/checkpoint.pth.tar'
# checkpoint_foot='https://download.openmmlab.com/mmpose/plugin/smoothnet/smoothnet_ws16_h36m.pth'

def main(pklfile):
    filter_cfg = dict(
        type='SmoothNetFilter',
        window_size=64,
        output_size=64,
        checkpoint=checkpoint,
        hidden_size=512,
        res_hidden_size=256,
        num_blocks=3,
        root_index=None)

    filter = build_filter(filter_cfg)
    filter_cfg_foot = dict(
        type='SmoothNetFilter',
        window_size=16,
        output_size=16,
        checkpoint=checkpoint_foot,
        hidden_size=512,
        res_hidden_size=256,
        num_blocks=3,
        root_index=None)
    filter_foot = build_filter(filter_cfg_foot)

    # %%
    pkldata=pickle.load(open(pklfile, 'rb'))
    calibPredict=CalibPredict(pkldata)
    X_global = pkldata['keypoints_xyz_ba']
    coms_3d = pkldata['extra']['coms_3d'][:, :, None, :]
    p_max = pkldata['extra']['keypoints_xyz_pmax']
    X_centored = X_global - coms_3d

    X_centored = X_centored[:].astype(np.float32)
    centor = coms_3d[:].astype(np.float32)
    # %%
    def filter_xyz(X, filter):
        T, N, K, D = X.shape
        X_re = X.reshape(T, N*K, D)
        X_smoothed = filter(X_re)
        X_smoothed = X_smoothed.reshape(T, N, K, D)
        return X_smoothed

    foot_KPT_index = [7,9,11,13]
    X_smoothed = filter_xyz(X_centored, filter)
    X_smoothed_foot = filter_xyz(np.ascontiguousarray(X_centored[:,:,foot_KPT_index,:]), filter_foot)
    X_smoothed[:,:,foot_KPT_index,:] = X_smoothed_foot

    X_global_smoothed = X_smoothed + centor

    X_global_smoothed_xy = calibPredict.p3d_to_p2d(X_global_smoothed)
    outdict = {**pkldata, 'keypoints_xyz_ba': X_global_smoothed.astype(np.float16),
                'keypoints_xy_ba': X_global_smoothed_xy.astype(np.float16)}

    outpklfile =  pklfile.replace('.matcalibpkl', '.smoothed_foot.matcalibpkl')
    pickle.dump(outdict, open(outpklfile, 'wb'))
    print('Saved to', outpklfile)
    # print('Generating videos...')
    # from lilab.mmpose.s3_matcalibpkl_2_video2d import main
    # main(outpklfile, 1, 'smoothed_foot')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pklfile', type=str)
    args = parser.parse_args()
    main(args.pklfile)
