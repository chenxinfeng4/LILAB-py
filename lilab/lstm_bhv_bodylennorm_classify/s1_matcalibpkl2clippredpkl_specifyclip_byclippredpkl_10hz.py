# conda activate OpenLabCluster
# project_dir="/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/24FP-behCluster/2407FP-total"
#%%
import pickle
import pandas as pd
import numpy as np
import glob
import os
import os.path as osp
import tqdm
import sys
from lilab.lstm_bhv_bodylennorm_classify.s1_matcalibpkl2clippredpkl_semiseq2seq import (
    load_engine, predict_cluster_labels, torch, SemiSeq2Seq, SemiSeq2Seq_simple
)
from lilab.lstm_bhv_bodylennorm_classify.s1_matcalibpkl2clippredpkl import clippreddata_r
import scipy.special
import argparse
from lilab.lstm_bhv_bodylennorm_classify.s01_matcalibpkl2rawfeatpkl import inplace_calculate_feat_clips


#%%
def predict_cluster_labels(outpkl, trt_model, df_records, feat_clips):
    outlabels = []
    outpvalues = []
    feat_dec = []
    with torch.no_grad():
        for out_feature in tqdm.tqdm(feat_clips):
            out_feature_torch = (
                torch.from_numpy(out_feature[None]).cuda().float()
            )  # (1,24,32)
            out_label = trt_model(out_feature_torch)
            out_label = out_label.detach().cpu().numpy().squeeze()
            # softmax 
            ind_max = np.argmax(out_label)
            outpvalue = scipy.special.softmax(out_label)[ind_max]
            outlabels.append(ind_max)
            outpvalues.append(outpvalue)

    outlabels = np.array(outlabels) + 1
    outpvalues = np.array(outpvalues)
    feat_dec = np.array(feat_dec)
    # save to file
    df_records['cluster_labels'] = outlabels
    df_clipNames = df_records[['vnake', 'startFrame', 'isBlack', 'cluster_labels']].copy()
    nsample = len(df_clipNames)
    df_clipNames['cluster_pvalues'] = outpvalues
    df_clipNames['startTime'] = df_clipNames['startFrame'] / 30
    project_dir = osp.dirname(outpkl)
    os.makedirs(project_dir, exist_ok=True)
    pickle.dump({'df_clipNames':df_clipNames, 
                'embedding': np.ones((nsample, 12)) + np.nan,
                'embedding_d2': np.ones((nsample, 2)) + np.nan,
                'cluster_labels': outlabels,
                'cluster_pvalues': outpvalues,
                'ncluster': clippreddata_r['ncluster'],
                'cluster_names': clippreddata_r['cluster_names'],
                'nK_mutual': clippreddata_r['nK_mutual'],
                'ntwin': clippreddata_r['ntwin'],
                'clipNames': np.array(['']*nsample),
                'nK_mirror_half': clippreddata_r['nK_mirror_half'],
                'cluster_names_mutualmerge': clippreddata_r['cluster_names_mutualmerge']},
                open(outpkl, 'wb'))


def main(project_dir, use_normed=False):
    trt_model = load_engine(use_normed)
    
    pklfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/24FP-behCluster/2407FP-total/densepredict_photometry/clipNames.clippredpkl'
    df_records = pickle.load(open(pklfile, 'rb'))['df_clipNames']
    if use_normed:
        rawfeatpkl_file = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/24FP-behCluster/2407FP-total/rawfeat_norm.pkl'
    else:
        rawfeatpkl_file = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/24FP-behCluster/2407FP-total/rawfeat.pkl'
    print('loading rawfeat from', rawfeatpkl_file)
    feat_dict = pickle.load(open(rawfeatpkl_file, 'rb'))

    feat_clips = inplace_calculate_feat_clips(df_records, feat_dict)
        
    outpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/24FP-behCluster/2407FP-total/densepredict_photometry_norm/densepredict.clippredpkl'
    predict_cluster_labels(outpkl, trt_model, df_records, feat_clips)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_dir', type=str)
    parser.add_argument('--use-normed', action='store_true')
    args = parser.parse_args()
    main(args.project_dir, args.use_normed)
