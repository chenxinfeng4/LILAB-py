#%%
import pickle
import numpy as np
import os.path as osp
import glob
import tqdm
import pandas as pd
from lilab.lstm_bhv_bodylennorm_classify.utilities_package_feature import package_feature
import torch
torch.zeros((3,3)).cuda()
import argparse
import multiprocessing
import scipy.special
import os


clippredfile_ref = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day55_Mix_analysis/SexAgeDay55andzzcWTinAUT_MMFF/result32/representitive_k36_filt_perc65/Representive_K36.clippredpkl'
clippreddata_r = pickle.load(open(clippredfile_ref, 'rb'))
engine = '/home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/lstm_bhv_bodylennorm_classify/lstm_behavior_offline.engine'

# project_dir = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day55_Mix_analysis/SexAgeDay55andzzcWTinAUT_MMFF/data'
clip_length = 24
maxlen = 27000

# load the dataset
def load(project_dir):
    bodylength_pkl = osp.join(project_dir, 'bodylength.pkl')
    bodylength_data = pickle.load(open(bodylength_pkl, 'rb'))

    bodylength_l = bodylength_data['bodylength_l']
    sniff_zoom_l = bodylength_data['sniff_zoom_l']
    matcalibpkl_nake_l = bodylength_data['matcalibpkl_nake_l']

    matcalibpkl_l = glob.glob(osp.join(project_dir+'/*.smooth*.matcalibpkl'))
    vnake_l = [osp.basename(osp.splitext(f)[0]).split('.')[0] for f in matcalibpkl_l]
    assert set(vnake_l) <= set(matcalibpkl_nake_l)

    paramdict = {f:(bodylength_l[i], sniff_zoom_l[i]) for i, f in enumerate(matcalibpkl_nake_l)}
    return matcalibpkl_l, vnake_l, paramdict


# create the clipNames
def create_clips(matcalibpkl_l, vnake_l, paramdict):
    df_records = pd.DataFrame()
    for v, matcalibpkl in zip(tqdm.tqdm(vnake_l), matcalibpkl_l):
        p3d = pickle.load(open(matcalibpkl, 'rb'))['keypoints_xyz_ba']
        p3d_CSK3 = p3d.transpose([1,0,2,3]).astype(float)
        p3d_CSK3 = p3d_CSK3[:,:maxlen]
        p3d_CSK3 = p3d_CSK3[:,:int(p3d_CSK3.shape[1]//clip_length*clip_length)]
        startFrames = np.arange(0, p3d_CSK3.shape[1], clip_length)
        vnake = [v] * len(startFrames) * 2
        isBlack = [True] * len(startFrames) + [False] * len(startFrames)
        startFrame = np.concatenate([startFrames, startFrames])
        df_records = df_records.append(pd.DataFrame({'vnake': vnake, 'startFrame': startFrame, 'isBlack': isBlack}), ignore_index=True)
    
    matdict = {}
    for v, matcalibpkl in zip(tqdm.tqdm(vnake_l), matcalibpkl_l):
        p3d = pickle.load(open(matcalibpkl, 'rb'))['keypoints_xyz_ba']
        p3d_CSK3 = p3d.transpose([1,0,2,3]).astype(float)
        p3d_CSK3 = p3d_CSK3[:,:maxlen]
        body_length, sniff_zoom_length = paramdict[v]
        matdict[v] = {'body_length': body_length, 'sniff_zoom_length': sniff_zoom_length, 'p3d_CSK3': p3d_CSK3}

        df_ind = df_records['vnake']==v
        df_records.loc[df_ind, 'body_length'] = body_length
        df_records.loc[df_ind, 'sniff_zoom_length'] = sniff_zoom_length
    return df_records, matdict


def worker(args):
    matcalibpkl_nake, matcalibpkl, paramdict = args
    feat_l = dict()
    bodylength, sniff_zoom = paramdict[matcalibpkl_nake]
    p3d_CTK3 = pickle.load(open(matcalibpkl, 'rb'))['keypoints_xyz_ba'].transpose([1,0,2,3]).astype(float)
    feat_black_first = package_feature(p3d_CTK3, bodylength, sniff_zoom)
    feat_white_first = package_feature(p3d_CTK3[::-1], bodylength, sniff_zoom)
    feat_l[matcalibpkl_nake + '_blackFirst'] = feat_black_first.T.astype(np.float32)  #Tx32
    feat_l[matcalibpkl_nake + '_whiteFirst'] = feat_white_first.T.astype(np.float32)
    return feat_l


def inplace_calculate_feat_clips(df_records, feat_dict):
    df_records['feat_clip_index'] = np.arange(len(df_records))
    feat_clips = np.zeros([df_records.shape[0], 24, next(iter(feat_dict.values())).shape[1]], dtype=np.float32)
    for i, record in enumerate(tqdm.tqdm(df_records.itertuples(), total=len(df_records))):
        feat_wave = feat_dict[record.vnake + ('_blackFirst' if record.isBlack else '_whiteFirst')]
        feat_clip = feat_wave[record.startFrame:record.startFrame+24]
        feat_clips[i] = feat_clip
    
    return feat_clips


# predict the cluster labels
def load_engine():
    from torch2trt import TRTModule
    trt_model = TRTModule()
    trt_model.load_from_engine(engine)
    input_shape = tuple(trt_model.context.get_binding_shape(0))
    input_numpy = np.random.rand(*input_shape).astype(np.float32)
    output = trt_model(torch.from_numpy(input_numpy).cuda().float())
    assert output.shape[1] == clippreddata_r['ncluster']
    return trt_model


def predict_cluster_labels(project_dir, trt_model, df_records, feat_clips):
    outlabels = []
    outpvalues = []
    outencs = []
    with torch.no_grad():
        for out_feature in tqdm.tqdm(feat_clips):
            out_feature_torch = (
                torch.from_numpy(out_feature[None]).cuda().float()
            )  # (1,24,32)
            output = trt_model(out_feature_torch)
            if len(output)==1:
                enc = np.zeros(12)
                out_label = output.detach().cpu().numpy().squeeze()
            else:
                enc, out_label = [x.detach().cpu().numpy().squeeze() for x in output]
            # softmax 
            ind_max = np.argmax(out_label)
            outpvalue = scipy.special.softmax(out_label)[ind_max]
            outlabels.append(ind_max)
            outpvalues.append(outpvalue)
            outencs.append(enc)

    outlabels = np.array(outlabels) + 1
    outpvalues = np.array(outpvalues)
    outencs = np.array(outencs)
    # save to file
    df_records['cluster_labels'] = outlabels
    df_clipNames = df_records[['vnake', 'startFrame', 'isBlack', 'cluster_labels']].copy()
    nsample = len(df_clipNames)
    os.makedirs(project_dir, exist_ok=True)
    outpkl = project_dir + '/lstm_offline.clippredpkl'
    pickle.dump({'df_clipNames':df_clipNames, 
                'embedding': outencs,
                'embedding_d2': np.zeros([nsample, 2]),
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
    print('save to', outpkl)
    

def main(project_dir):
    trt_model = load_engine()
    print('load engine done')

    matcalibpkl_l, vnake_l, paramdict = load(project_dir)
    df_records, matdict = create_clips(matcalibpkl_l, vnake_l, paramdict)

    args_l = [(v, m, paramdict) for v, m in zip(vnake_l, matcalibpkl_l)]
    with multiprocessing.Pool(processes=40) as pool:
        results = list(tqdm.tqdm(pool.imap_unordered(worker, args_l), total=len(matdict)))
    feat_dict = dict()
    for result in results:
        feat_dict.update(result)
    feat_clips = inplace_calculate_feat_clips(df_records, feat_dict)
    print('predicting')
    predict_cluster_labels(project_dir, trt_model, df_records, feat_clips)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_dir', type=str)
    args = parser.parse_args()
    main(args.project_dir)
