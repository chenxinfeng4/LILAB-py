# conda activate mmdet
# 速度索引为[0,1]
#%% 计算速度分位数
import pickle
import numpy as np
import pandas as pd
import os.path as osp
import matplotlib.pyplot as plt

clip_label_pkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day55_Mix_analysis/SexAgeDay55andzzcWTinAUT_MMFF/data/train_test_clip_newlabels_k36.pkl'
#%%
clip_label_data = pickle.load(open(clip_label_pkl, 'rb'))
df_records = clip_label_data['df_records']
feat_clip = np.array([f for f in df_records['feat_clip']])

speed_ind = [0, 1]
feat_clip_speed = feat_clip[:, speed_ind, :]


#%%
percentile = 95
feat_clip_speed_norm = np.percentile(feat_clip_speed.ravel(), percentile)
feat_clip_normed = feat_clip.copy()
feat_clip_normed[:, speed_ind, :] = feat_clip_speed / feat_clip_speed_norm
feat_clip_normed[:, speed_ind, :] = np.clip(feat_clip_normed[:, speed_ind, :], 0, 4)

df_records['feat_clip_norm'] = [*feat_clip_normed]
clip_label_data['df_records'] = df_records
clip_label_data['percentile_speed'] = [percentile, feat_clip_speed_norm]
pickle.dump(clip_label_data, open(clip_label_pkl, 'wb'))


plt.figure()
plt.subplot(1,2,1)
plt.hist(feat_clip_speed.flatten(), bins=100)
plt.xlabel('feature')
plt.ylabel('count')
plt.title('histogram of feature')
plt.subplot(1,2,2)
plt.hist(feat_clip_normed[:, speed_ind, :].flatten(), bins=100)
plt.xlabel('feature')
plt.ylabel('count')
plt.xlim([-0.2, 4.8])
plt.title('histogram of feature normed at 95%')

plt.savefig('/home/liying_lab/chenxf/ml-project/LILAB-py/lilab/lstm_bhv_bodylennorm_classify/feat_clip_train_speed_hist.jpg')
