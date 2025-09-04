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
    load, load_engine, predict_cluster_labels, SemiSeq2Seq, nn, torch
)
from lilab.lstm_bhv_bodylennorm_classify.s1_matcalibpkl2clippredpkl import clippreddata_r
import scipy.special
import argparse
from lilab.lstm_bhv_bodylennorm_classify.utilities_package_feature import package_feature
from lilab.OpenLabCluster_train.model import seq2seq


#%%
def create_clips(vnake_l, rawfeat_pkl):
    # pklfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/24FP-behCluster/2407FP-total/out_semiseq2seq/representitive_k36_filt_perc88/concatTime_moduleWise_pre2clipNotEqual/concatTime_moduleWisee_pre2clipNotEqual_shift_0.clippredpkl'
    pklfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/24FP-behCluster/2407FP-total/out_semiseq2seq/representitive_k36_filt_perc88/concatTime_moduleWise_pre2clipNotEqual_chasepin1clip/concatTime_moduleWisee_pre2clipNotEqual_chasepin1clip_shift_0.clippredpkl'
    with open(pklfile, 'rb') as f:
        pkldata = pickle.load(f)
    df_records = pkldata['df_clipNames']

    df_records['center_frame'] = df_records['startFrame']  # shift 0
    df_records['cluster_label'] = df_records['cluster_labels']
    df_records['center_cluster_label'] = df_records['cluster_label']
    assert set(df_records['vnake']).issubset(set(vnake_l))

    df_records_dense=[]
    for di in range(df_records.shape[0]):
        df=df_records.iloc[di]
        df['startFrame']=[df['startFrame']+i for i in np.arange(-12,13, 3)] #considering start real left and right
        dks=list(df.keys())
        df_records_dense.append(pd.DataFrame(dict(zip(dks,[df[dk] for dk in dks]))))
    df_records_dense=pd.concat(df_records_dense)
    df_records_dense = df_records_dense[df_records_dense['startFrame']>=0]
    df_records = df_records_dense

    df_records['feat_clip'] = None
    p3d_CSK3_clip_l = []
    feat_clip_l = []
    for record in tqdm.tqdm(df_records.itertuples(), total=len(df_records)):
        record.vnake, record.startFrame, record.isBlack
        p3d_CSK3 = matdict[record.vnake]['p3d_CSK3']

        feat_clip_l.append(feat_clip[None].astype(np.float32))
    
    feat_clips = np.concatenate(feat_clip_l).transpose((0,2,1)) #(n,24,32)
    return df_records, matdict, feat_clips


def predict_cluster_labels(project_dir, trt_model, df_records, feat_clips):
    outlabels = []
    outpvalues = []
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
    # save to file
    df_records['cluster_labels'] = outlabels
    df_clipNames = df_records[['vnake', 'startFrame', 'isBlack', 'cluster_labels']].copy()
    nsample = len(df_clipNames)
    df_clipNames['startTime'] = df_clipNames['startFrame'] / 30
    os.makedirs(project_dir, exist_ok=True)
    outpkl = project_dir + '/concatTime_moduleWise_pre2clipNotEqual_shift_0_densesample_k36.clippredpkl'
    pickle.dump({'df_clipNames':df_clipNames, 
                'embedding': np.zeros((nsample, 12))+np.nan,
                'embedding_d2': np.zeros((nsample, 2))+np.nan,
                'feat_clips': feat_clips,
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
    matcalibpkl_l, vnake_l, paramdict = load(project_dir)
    df_records, matdict, feat_clips = create_clips(matcalibpkl_l, vnake_l, paramdict)

    trt_model = load_engine()
    output_dir = osp.join(project_dir, 'specifyclip')
    predict_cluster_labels(output_dir, trt_model, df_records, feat_clips)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_dir', type=str)
    parser.add_argument('--use-normed', action='store_true')
    args = parser.parse_args()
    main(args.project_dir, args.use_normed)
