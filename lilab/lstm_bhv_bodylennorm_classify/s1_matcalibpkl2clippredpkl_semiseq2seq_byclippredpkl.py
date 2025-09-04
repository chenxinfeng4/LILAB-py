#%%
import pickle
import numpy as np
import argparse
import glob
from lilab.lstm_bhv_bodylennorm_classify.s1_matcalibpkl2clippredpkl import (
    load_engine, predict_cluster_labels
)
import torch
import torch.nn as nn
import os.path as osp
from openlabcluster.training_utils.ssl.SeqModel import SemiSeq2Seq
import pandas as pd
import tqdm
from lilab.lstm_bhv_bodylennorm_classify.s1_matcalibpkl2clippredpkl_semiseq2seq import (
    SemiSeq2Seq_simple, load_engine, clip_length
)


def create_clips(vnake_l, rawfeat_pkl, clippredpkl):
    rawfeat_data = pickle.load(open(rawfeat_pkl, 'rb'))

    keys = np.array([(v+'_blackFirst', v+'_whiteFirst') for v in vnake_l])
    assert set(keys.ravel()) == set(rawfeat_data.keys())

    df_clipNames = pickle.load(open(clippredpkl, 'rb'))['df_clipNames']
    df_records = pd.DataFrame()
    feat_NTK_l = []
    for v, (key_black, key_white) in zip(vnake_l, keys):
        startFrame_b = df_clipNames.loc[(df_clipNames.vnake==v)&(df_clipNames.isBlack==True), 'startFrame'].values
        startFrame_w = df_clipNames.loc[(df_clipNames.vnake==v)&(df_clipNames.isBlack==False), 'startFrame'].values
        startFrame_b, startFrame_w = np.sort(startFrame_b), np.sort(startFrame_w)
        startFrame = np.concatenate([startFrame_b, startFrame_w])
        isBlack = np.concatenate([np.ones(len(startFrame_b)), np.zeros(len(startFrame_w))])==1

        feat_NK_black = [rawfeat_data[key_black][s:s+clip_length] for s in startFrame_b]
        feat_NK_white = [rawfeat_data[key_white][s:s+clip_length] for s in startFrame_w]
        feat_NTK_l.extend([feat_NK_black, feat_NK_white])

        df_records = df_records.append(pd.DataFrame({'vnake': v, 'startFrame': startFrame, 'isBlack': isBlack}), ignore_index=True)
    
    feat_NTK = np.concatenate(feat_NTK_l, axis=0)
    df_records['feat_clip_index'] = np.arange(len(df_records))

    return df_records, feat_NTK


def main(project_dir, clippredpkl, use_normed=False):
    vnake_l = [osp.basename(osp.splitext(f)[0]).split('.')[0] 
               for f in glob.glob(osp.join(project_dir+'/*.smooth*.matcalibpkl'))]
    rawfeat_pkl = osp.join(project_dir, 'rawfeat_norm.pkl' if use_normed else 'rawfeat.pkl')
    print('====='*15)
    print('rawfeat_pkl:', rawfeat_pkl)
    df_records, feat_clips = create_clips(vnake_l, rawfeat_pkl, clippredpkl)
    
    trt_model = load_engine(use_normed)
    project_dir = osp.join(project_dir, 'out_semiseq2seq_byclip' + ('_norm' if use_normed else ''))
    predict_cluster_labels(project_dir, trt_model, df_records, feat_clips)
    print('output_dir:', project_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_dir', type=str)
    parser.add_argument('clippredpkl', type=str)
    parser.add_argument('--use-normed', action='store_true')
    args = parser.parse_args()
    main(args.project_dir, args.clippredpkl, args.use_normed)
