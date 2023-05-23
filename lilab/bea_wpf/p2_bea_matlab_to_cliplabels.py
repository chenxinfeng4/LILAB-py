# python -m lilab.bea_wpf.p2_bea_matlab_to_cliplabels matfile sampleinfopkl
#%%        
import numpy as np
import pickle
import os.path as osp
import pandas as pd
import scipy.io as sio
import argparse

matfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zzc_to_dzy/f2m2_veh/2023-02-25_15-15-02_a2p4_merge_coords3d_BeA_Vel.mat'
sampleinfopkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zzc_to_dzy/f2m2_veh/2023-02-25_15-15-02_a2p4_merge_samplesinfo.pkl'

# %%
def main(matfile, sampleinfopkl):
    matdata = sio.loadmat(matfile)
    label_seq = np.squeeze(matdata['label_seq'])
    # %%
    # 'video_nake_name', 'frameid', 'label'

    datapkl = pickle.load(open(sampleinfopkl, 'rb'))
    nsample_l = datapkl['nsample_l']
    if 'downsample_rate' in datapkl:
        downsample_rate = datapkl['downsample_rate']
        nframe = sum(ns[1] for ns in nsample_l)
        assert np.ceil(nframe / downsample_rate) == label_seq.shape[0], 'Not equal length'
        label_seq_full = np.zeros((len(label_seq)*downsample_rate, ))
        for i in range(len(label_seq)):
            label_seq_full[i*downsample_rate:(i+1)*downsample_rate] = label_seq[i]
        label_seq = label_seq_full[:nframe]

    video_nake_name_l = []
    frameid_l = []

    for f, nframe in nsample_l:
        video_nake_name = osp.basename(f).split('.')[0]
        video_nake_name_l.extend([video_nake_name]*nframe)
        frameid_l.extend(list(range(nframe)))
    assert len(frameid_l) == label_seq.shape[0], 'Not equal length'

    df_clip = pd.DataFrame({'video_nake_name':video_nake_name_l, 'frameid':frameid_l, 'label':label_seq})

    outpkl = osp.splitext(matfile)[0] + '.clip_framebyframe_labels.pkl'
    outdict = {'df_clip': df_clip}
    pickle.dump(outdict, open(outpkl, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('matfile', type=str)
    parser.add_argument('sampleinfopkl', type=str)
    args = parser.parse_args()
    main(args.matfile, args.sampleinfopkl)
