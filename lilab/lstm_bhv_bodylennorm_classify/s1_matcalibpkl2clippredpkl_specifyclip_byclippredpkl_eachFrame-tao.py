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


class SemiSeq2Seq_simple(nn.Module):
    def __init__(self, model:SemiSeq2Seq):
        super().__init__()
        self.model = model
    
    def __forward(self, input_tensor):
        seq_len = [24] * input_tensor.shape[0]
        outs = self.model(input_tensor, seq_len)
        return outs
    
    def forward(self, input_tensor):
        self.model.seq.__class__ = seq2seq
        seq_len = [24] * input_tensor.shape[0]
        _, deout, deout_seq_ = self.model.seq.forward_test(input_tensor, seq_len)
        *_, pred  = self.model(input_tensor, seq_len)
        return pred, deout_seq_


def load_engine():
    model_file = '/home/liying_lab/chenxf/ml-project/LILAB-py/lilab/lstm_bhv_bodylennorm_classify/semiseq2seq_behavior_offline.pt'
    input_tensor = torch.randn(1, 24, 32).cuda().float()
    model = torch.load(model_file)
    model = model.cuda()
    output = model(input_tensor)
    return model

#%%
def create_clips(matcalibpkl_l, vnake_l, paramdict):
    pklfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/24FP-behCluster/2407FP-total/out_semiseq2seq/representitive_k36_filt_perc88/concatTime_moduleWise/concatTime_moduleWise_shift_04.clippredpkl'
    
    with open(pklfile, 'rb') as f:
        pkldata = pickle.load(f)
    df_records = pkldata['df_clipNames']

    df_records['center_frame'] = df_records['startFrame'] + 12
    df_records['center_cluster_label'] = df_records['cluster_label']
    assert set(df_records['vnake']).issubset(set(vnake_l))

    matdict = {}
    for v, matcalibpkl in zip(tqdm.tqdm(vnake_l), matcalibpkl_l):
        p3d = pickle.load(open(matcalibpkl, 'rb'))['keypoints_xyz_ba']
        p3d_CSK3 = p3d.transpose([1,0,2,3]).astype(float)
        body_length, sniff_zoom_length = paramdict[v]
        matdict[v] = {'body_length': body_length, 'sniff_zoom_length': sniff_zoom_length, 'p3d_CSK3': p3d_CSK3}
        df_ind = df_records['vnake']==v
        df_records.loc[df_ind, 'body_length'] = body_length
        df_records.loc[df_ind, 'sniff_zoom_length'] = sniff_zoom_length
    
    df_records['p3d_CSK3_clip'] = None
    df_records['feat_clip'] = None
    for record in tqdm.tqdm(df_records.itertuples(), total=len(df_records)):
        record.vnake, record.startFrame, record.isBlack
        p3d_CSK3 = matdict[record.vnake]['p3d_CSK3']
        p3d_CSK3_clip = p3d_CSK3[[0,1] if record.isBlack else [1,0], record.startFrame:record.startFrame+24]
        feat_clip = package_feature(p3d_CSK3_clip, record.body_length, record.sniff_zoom_length)
        df_records.loc[record.Index, 'p3d_CSK3_clip'] = p3d_CSK3_clip[None].astype(np.float32)
        df_records.loc[record.Index, 'feat_clip'] = feat_clip[None].astype(np.float32)

    return df_records, matdict



def predict_cluster_labels(project_dir, trt_model, df_records, feat_clips):
    outlabels = []
    outpvalues = []
    feat_dec = []
    with torch.no_grad():
        for out_feature in tqdm.tqdm(feat_clips):
            out_feature_torch = (
                torch.from_numpy(out_feature[None]).cuda().float()
            )  # (1,24,32)
            out_label, feat_24 = trt_model(out_feature_torch)
            feat_24 = feat_24.detach().cpu().numpy().squeeze()
            out_label = out_label.detach().cpu().numpy().squeeze()
            # softmax 
            ind_max = np.argmax(out_label)
            outpvalue = scipy.special.softmax(out_label)[ind_max]
            outlabels.append(ind_max)
            outpvalues.append(outpvalue)
            feat_dec.append(feat_24)

    outlabels = np.array(outlabels) + 1
    outpvalues = np.array(outpvalues)
    feat_dec = np.array(feat_dec)
    # save to file
    df_records['cluster_labels'] = outlabels
    df_clipNames = df_records[['vnake', 'startFrame', 'isBlack', 'cluster_labels']].copy()
    nsample = len(df_clipNames)
    df_clipNames['feat_dec'] = [f for f in feat_dec.astype(float)]
    df_clipNames['startTime'] = df_clipNames['startFrame'] / 30
    os.makedirs(project_dir, exist_ok=True)
    outpkl = project_dir + '/concatTime_moduleWise_shift_04_decseq.clippredpkl'
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


def main(project_dir):
    matcalibpkl_l, vnake_l, paramdict = load(project_dir)
    df_records, matdict = create_clips(matcalibpkl_l, vnake_l, paramdict)

    feat_clips = np.array([f for f in df_records['feat_clip']]).transpose((0,2,1)) #(n,24,32)
        
    trt_model = load_engine()
    output_dir = osp.join(project_dir, 'specifyclip')
    predict_cluster_labels(output_dir, trt_model, df_records, feat_clips)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_dir', type=str)
    args = parser.parse_args()
    main(args.project_dir)
