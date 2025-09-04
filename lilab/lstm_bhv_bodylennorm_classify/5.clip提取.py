#%%
import pickle
import pandas as pd
import numpy as np
import glob
import os.path as osp
import tqdm
from lilab.openlabcluster_postprocess.s1_merge_3_file import get_assert_1_file
import sys

from lilab.lstm_bhv_bodylennorm_classify.utilities_package_feature import package_feature

project_dir_offline = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day55_Mix_analysis/SexAgeDay55andzzcWTinAUT_MMFF/data'
project_dir = project_dir_offline
# repr_filt_dir = '/DATA/taoxianming/rat/data/Mix_analysis/SexAgeDay55andzzcWTinAUT_MMFF/result32/semiseq2seq_iter0/output/far_ns_with_s_recluster_k34/'
# clippred_pkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day55_Mix_analysis/SexAgeDay55andzzcWTinAUT_MMFF/result32/FWPCA0.00_P100_en3_hid30_epoch264-decSeqPC0.9_svm2allAcc0.94_kmeansK2use-42_fromK1-20_K100.afterALclippredpkl'
# clippred_data = pickle.load(open(clippred_pkl, 'rb'))

clip_label_pkl = project_dir +'/train_test_clip_newlabels_k36.pkl'
clip_label_data = pickle.load(open(clip_label_pkl, 'rb'))
# df_clipNames = clippred_data['df_clipNames']
# df_clipNames['embedding'] = object
# for i in range(len(df_clipNames['embedding'])):
#     df_clipNames['embedding'].iloc[i] = clippred_data['embedding'][i]
"""
这是Day 55天的训练 semiseq2seq 标签预测模型
最重要的就是这个 df_records,
- 包含了所有clip的3D坐标点信息 p3d_CSK3_clip
- clip的32类特征信息 feat_clip
- 必要的normalize参数 body_length, sniff_zoom_length
- 包括训练集、测试集的拆分信息
"""
df_records = clip_label_data['df_records']
df_records['p3d_CSK3_clip'] = object
df_records['feat_clip'] = object
df_records['body_length'] = np.nan
df_records['sniff_zoom_length'] = np.nan

# df_clipNames[['vnake', 'isBlack', 'startFrame']].values == df_records[['vnake', 'isBlack', 'startFrame']].values


matcalibpkl_l = glob.glob(osp.join(project_dir_offline+'/*.smooth*.matcalibpkl'))
vnake_l = [osp.basename(osp.splitext(f)[0]).split('.')[0] for f in matcalibpkl_l]
matcalibpkl_l = [get_assert_1_file(osp.join(project_dir, f'{v}.smooth*.matcalibpkl')) for v in vnake_l]


bodylength_pkl = osp.join(project_dir_offline, 'bodylength.pkl')
bodylength_data = pickle.load(open(bodylength_pkl, 'rb'))

bodylength_l = bodylength_data['bodylength_l']
sniff_zoom_l = bodylength_data['sniff_zoom_l']
matcalibpkl_nake_l = bodylength_data['matcalibpkl_nake_l']


assert set(vnake_l) == set(df_records['vnake']) and set(vnake_l) == set(matcalibpkl_nake_l)

paramdict = {f:(bodylength_l[i], sniff_zoom_l[i]) for i, f in enumerate(matcalibpkl_nake_l)}

matdict = {}
for v, matcalibpkl in zip(tqdm.tqdm(vnake_l), matcalibpkl_l):
    p3d = pickle.load(open(matcalibpkl, 'rb'))['keypoints_xyz_ba']
    p3d_CSK3 = p3d.transpose([1,0,2,3]).astype(float)
    body_length, sniff_zoom_length = paramdict[v]
    matdict[v] = {'body_length': body_length, 'sniff_zoom_length': sniff_zoom_length, 'p3d_CSK3': p3d_CSK3}
    df_ind = df_records['vnake']==v
    df_records.loc[df_ind, 'body_length'] = body_length
    df_records.loc[df_ind, 'sniff_zoom_length'] = sniff_zoom_length

#%%
for record in tqdm.tqdm(df_records.itertuples(), total=len(df_records)):
    record.vnake, record.startFrame, record.isBlack
    p3d_CSK3 = matdict[record.vnake]['p3d_CSK3']
    p3d_CSK3_clip = p3d_CSK3[[0,1] if record.isBlack else [1,0], record.startFrame:record.startFrame+24]
    feat_clip = package_feature(p3d_CSK3_clip, record.body_length, record.sniff_zoom_length)
    df_records.loc[record.Index, 'p3d_CSK3_clip'] = p3d_CSK3_clip[None].astype(np.float32)
    df_records.loc[record.Index, 'feat_clip'] = feat_clip[None].astype(np.float32)

# df_records2 = pd.merge(df_records, df_clipNames[['vnake', 'isBlack', 'startFrame', 'embedding']],
#         on=['vnake', 'isBlack', 'startFrame'], how='inner')

clip_label_data['df_records'] = df_records
# clip_label_data['df_records2'] = df_records2
out_pkl = osp.join(project_dir_offline, osp.basename(clip_label_pkl))
pickle.dump(clip_label_data, open(out_pkl, 'wb'))
