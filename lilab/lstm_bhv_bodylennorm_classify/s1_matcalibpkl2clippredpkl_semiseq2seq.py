#%%
import pickle
import matplotlib
matplotlib.use('Agg')

import numpy as np
from lilab.lstm_bhv_bodylennorm_classify.utilities_package_feature import package_feature
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


clip_length = 24
#%%
class SemiSeq2Seq_simple(nn.Module):
    def __init__(self, model:SemiSeq2Seq):
        super().__init__()
        self.model = model
    
    def forward(self, input_tensor):
        seq_len = [24] * input_tensor.shape[0]
        enc, *_, pred = self.model(input_tensor, seq_len)
        return enc, pred


def load_engine(use_normed=False):
    if use_normed:
        model_file = '/home/liying_lab/chenxf/ml-project/LILAB-py/lilab/lstm_bhv_bodylennorm_classify/model_percDepent/semiseq2seq_behavior_offline.pt'
    else:
        model_file = '/home/liying_lab/chenxf/ml-project/LILAB-py/lilab/lstm_bhv_bodylennorm_classify/semiseq2seq_behavior_offline.pt'
    input_tensor = torch.randn(1, 24, 32).cuda().float()
    model = torch.load(model_file)
    model = model.cuda()
    enc, output = model(input_tensor)
    print('load model:', model_file)
    return model


def create_clips(vnake_l, rawfeat_pkl, stride):
    rawfeat_data = pickle.load(open(rawfeat_pkl, 'rb'))

    maxlen = 27000
    keys = np.array([(v+'_blackFirst', v+'_whiteFirst') for v in vnake_l])
    assert set(keys.ravel()) == set(rawfeat_data.keys())

    df_records = pd.DataFrame()
    feat_NTK_l = []
    for v, (key_black, key_white) in zip(vnake_l, keys):
        feat_NK = rawfeat_data[key_black]
        N = min(feat_NK.shape[0], maxlen) // clip_length * clip_length
        startFrames = np.arange(0, N-clip_length+1, stride)
        vnake = [v] * len(startFrames) * 2
        startFrame = np.concatenate([startFrames, startFrames])
        isBlack = [True] * len(startFrames) + [False] * len(startFrames)

        feat_NK_black = [rawfeat_data[key_black][s:s+clip_length] for s in startFrames]
        feat_NK_white = [rawfeat_data[key_white][s:s+clip_length] for s in startFrames]
        feat_NTK_l.extend([feat_NK_black, feat_NK_white])
        # feat_NK_black = rawfeat_data[key_black][:N]
        # feat_NK_white = rawfeat_data[key_white][:N]
        # feat_NK = np.concatenate([feat_NK_black, feat_NK_white], axis=0)
        # feat_NTK = feat_NK.reshape(-1, clip_length, feat_NK.shape[-1])
        # feat_NTK_l.append(feat_NTK)
        df_records = df_records.append(pd.DataFrame({'vnake': vnake, 'startFrame': startFrame, 'isBlack': isBlack}), ignore_index=True)
    
    feat_NTK = np.concatenate(feat_NTK_l, axis=0)
    df_records['feat_clip_index'] = np.arange(len(df_records))

    return df_records, feat_NTK


def main(project_dir, use_normed=False, stride=clip_length):
    print('use_normed',use_normed) #zyq
    vnake_l = [osp.basename(osp.splitext(f)[0]).split('.')[0] 
               for f in glob.glob(osp.join(project_dir+'/*.smooth*.matcalibpkl'))]
    rawfeat_pkl = osp.join(project_dir, 'rawfeat_norm.pkl' if use_normed else 'rawfeat.pkl')
    print('====='*15)
    print('rawfeat_pkl:', rawfeat_pkl)
    df_records, feat_clips = create_clips(vnake_l, rawfeat_pkl, stride)
    
    trt_model = load_engine(use_normed)
    project_dir = osp.join(project_dir, 'out_semiseq2seq' + ('_norm' if use_normed else ''))
    predict_cluster_labels(project_dir, trt_model, df_records, feat_clips)
    print('output_dir:', project_dir)

# project_dir='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/24behv_s3ms/shank3'
# use_normed=False
# stride=clip_length
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_dir', type=str)
    parser.add_argument('--use-normed', action='store_true')
    parser.add_argument('--stride', type=int, default=clip_length)
    args = parser.parse_args()
    main(args.project_dir, args.use_normed, args.stride)
