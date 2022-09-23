# conda activate mmpose
# python -m lilab.smoothnet.s1_matcalibpkl2smooth xxx.matcalibpkl
# %% imports
import pickle
from mmpose.core.post_processing.temporal_filters import build_filter
from lilab.multiview_scripts_dev.s6_calibpkl_predict import CalibPredict
import argparse
import scipy.io as sio
# %%
pklfile='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-26_19-31-21_SHANK20_HetxHet.matcalibpkl'
# checkpoint='https://download.openmmlab.com/mmpose/plugin/smoothnet/smoothnet_ws8_h36m.pth'
# checkpoint='https://download.openmmlab.com/mmpose/plugin/smoothnet/smoothnet_ws64_h36m.pth'
checkpoint='/home/liying_lab/chenxinfeng/DATA/SmoothNet/results/h36m_cxf_16b3/checkpoint.pth.tar'
slide_window_size = 16
def main(pklfile):
    filter_cfg = dict(
        type='SmoothNetFilter',
        window_size=slide_window_size,
        output_size=slide_window_size,
        checkpoint=checkpoint,
        hidden_size=512,
        res_hidden_size=256,
        num_blocks=3,
        root_index=None)

    filter = build_filter(filter_cfg)

    # %%
    pkldata=pickle.load(open(pklfile, 'rb'))
    calibPredict=CalibPredict(pkldata)
    X_global = pkldata['keypoints_xyz_ba']
    coms_3d = pkldata['extra']['coms_3d'][:, :, None, :]
    p_max = pkldata['extra']['keypoints_xyz_pmax']
    X_centored = X_global - coms_3d

    X_centored = X_centored[:]
    centor = coms_3d[:]
    # %%
    T, N, K, D = X_centored.shape
    X = X_centored.reshape(T, N*K, D)
    X_smoothed = filter(X)
    X_smoothed = X_smoothed.reshape(T, N, K, D)
    X_global_smoothed = X_smoothed + centor

    X_global_smoothed_xy = calibPredict.p3d_to_p2d(X_global_smoothed)
    outdict = {**pkldata, 'keypoints_xyz_ba': X_global_smoothed,
                'keypoints_xy_ba': X_global_smoothed_xy}

    outpklfile =  pklfile.replace('.matcalibpkl', f'.smoothed_w{slide_window_size}.matcalibpkl')
    pickle.dump(outdict, open(outpklfile, 'wb'))
    sio.savemat(outpklfile.replace('.matcalibpkl', f'.smoothed_w{slide_window_size}.mat'), outdict)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pklfile', type=str)
    args = parser.parse_args()
    main(args.pklfile)
