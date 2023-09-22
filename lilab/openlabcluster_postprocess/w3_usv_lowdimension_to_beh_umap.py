# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from lilab.openlabcluster_postprocess.s1_merge_3_file import get_assert_1_file
import pandas as pd
from lilab.openlabcluster_postprocess.v2c_usv_tsne_lowfeature_scatter import get_norm
# %%
project_dir = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/DayAll20230828'
beh_dir = osp.join(project_dir, 'All-DecSeq')
beh_pkl = get_assert_1_file(osp.join(beh_dir, '*.clippredpkl'))
usv_dir = osp.join(project_dir, 'usv_cluster')
usv_pkl = get_assert_1_file(osp.join(usv_dir, '*.usvclippredpkl'))
usv_evt_pkl = get_assert_1_file(osp.join(usv_dir, 'usv_evt.usvpkl'))

#%%
beh_data = pickle.load(open(beh_pkl, 'rb'))
usv_data = pickle.load(open(usv_pkl, 'rb'))
usv_evt_data = pickle.load(open(usv_evt_pkl, 'rb'))

usvclipNames0 = usv_evt_data['df_usv_evt'][['video_nake', 'idx_in_file']]
usvclipNames  = pd.DataFrame(usv_data['clipNames'], columns=['video_nake', 'idx_in_file'])
assert np.all(usvclipNames.values==usvclipNames0.values)
video_nake = np.sort(usvclipNames['video_nake'].unique())
df_usv_evt = usv_evt_data['df_usv_evt']
df_usv_evt['center_t'] = (df_usv_evt['start'] + df_usv_evt['end'])/2
df_usv_evt['cluster_labels'] = usv_data['cluster_labels']
# df_usv_evt_sort = df_usv_evt.sort_values(by=['video_nake', 'idx_in_file'])
df_usv_evt_sort = df_usv_evt
#%%
behclipNames_ = beh_data['clipNames']
behclipNames_list = []
for c in behclipNames_:
    c_list = c.split('.')
    video_key = c_list[0]
    startFr = c_list[3]
    isblackFirst = 'blackFirst' in startFr
    frameid = int(startFr.split('startFrameReal')[-1])
    behclipNames_list.append((video_key, frameid, isblackFirst))

behclipNames = pd.DataFrame(behclipNames_list, columns=['video_nake', 'idx_in_file', 'is_black_first'])
# behclipNames_sort = behclipNames.sort_values(by=['video_nake', 'is_black_first', 'idx_in_file'])
behclipNames_sort = behclipNames
behclipNames_sort = behclipNames_sort[behclipNames_sort['video_nake'].isin(video_nake)]
behclipNames_sort['center_t'] = behclipNames_sort['idx_in_file'] / 30 + 0.4
#%%
# 开始做 match
index_in_behclip_bw_all = []
for video_key in video_nake:
    df_usv_evt_vkey = df_usv_evt_sort[df_usv_evt_sort['video_nake'] == video_key]
    behclipNames_vkey_bw = behclipNames_sort[behclipNames_sort['video_nake'] == video_key]
    behclipNames_vkey_b = behclipNames_vkey_bw[behclipNames_vkey_bw['is_black_first']]
    behclipNames_vkey_w = behclipNames_vkey_bw[~behclipNames_vkey_bw['is_black_first']]
    index_in_behclip_bw = []
    for behclipNames_vkey in [behclipNames_vkey_b, behclipNames_vkey_w]:
        n_usv_evt = df_usv_evt_vkey.shape[0]
        n_beh_evt = behclipNames_vkey.shape[0]

        dt_matrix = df_usv_evt_vkey['center_t'].values[:,None] - behclipNames_vkey['center_t'].values[None,:] #n_usv x n_beh
        dt_matrix_abs = np.abs(dt_matrix)
        dt_ind = np.array([np.argmin(m) for m in dt_matrix_abs])
        index_in_behclip = behclipNames_vkey.index[dt_ind]
        index_in_behclip_bw.append(index_in_behclip)
    index_in_behclip_bw = np.array(index_in_behclip_bw)
    index_in_behclip_bw_all.append(index_in_behclip_bw)

index_in_behclip_bw_all = np.concatenate(index_in_behclip_bw_all, axis=-1)

# %%
embedding_bw_d2 = beh_data['embedding_d2']
embedding_b_d2 = beh_data['embedding_d2'][index_in_behclip_bw_all[0]]
embedding_w_d2 = beh_data['embedding_d2'][index_in_behclip_bw_all[1]]
assert len(embedding_w_d2) == len(df_usv_evt)
df_usv_evt['embedding_w_d2_0'] = embedding_w_d2[:,0] + np.random.randn(len(embedding_w_d2))
df_usv_evt['embedding_w_d2_1'] = embedding_w_d2[:,1] + np.random.randn(len(embedding_w_d2))
df_usv_evt['embedding_b_d2_0'] = embedding_b_d2[:,0] + np.random.randn(len(embedding_w_d2))
df_usv_evt['embedding_b_d2_1'] = embedding_b_d2[:,1] + np.random.randn(len(embedding_w_d2))
df_usv_evt['main_freq_norm'] = get_norm(df_usv_evt['main_freq'])
df_usv_evt['duration_norm'] = get_norm(df_usv_evt['duration'])
df_usv_evt['distribution_norm'] = get_norm(df_usv_evt['duration'])

df_usv_evt_cp1 = df_usv_evt.copy()
df_usv_evt_cp2 = df_usv_evt.copy()
df_usv_evt_cp1['embedding_d2_0'] = df_usv_evt['embedding_b_d2_0']
df_usv_evt_cp1['embedding_d2_1'] = df_usv_evt['embedding_b_d2_1']
df_usv_evt_cp2['embedding_d2_0'] = df_usv_evt['embedding_w_d2_0']
df_usv_evt_cp2['embedding_d2_1'] = df_usv_evt['embedding_w_d2_1']

df_usv_evt_cp = pd.concat((df_usv_evt_cp1, df_usv_evt_cp2)).sample(frac=1)

axis_lim = [embedding_bw_d2.min(), embedding_bw_d2.max(), embedding_bw_d2.min(), embedding_bw_d2.max()]
nskip = 4
plt.figure(figsize=(10, 10))
# plt.subplot(211)
plt.scatter(df_usv_evt['embedding_w_d2_0'][::nskip], 
            df_usv_evt['embedding_w_d2_1'][::nskip],
            c = df_usv_evt['distribution_norm'][::nskip],
            s = 5,
            cmap = 'coolwarm')
# plt.axis(axis_lim)
plt.gca().set_aspect('equal', 'box')
# plt.subplot(212)
plt.figure(figsize=(10, 10))
plt.scatter(df_usv_evt['embedding_b_d2_0'][::nskip], 
            df_usv_evt['embedding_b_d2_1'][::nskip],
            c = df_usv_evt['main_freq_norm'][::nskip],
            s = 5,
            cmap = 'coolwarm')


plt.scatter(df_usv_evt['embedding_w_d2_0'][::nskip], 
            df_usv_evt['embedding_w_d2_1'][::nskip],
            c = df_usv_evt['duration_norm'][::nskip],
            cmap = 'coolwarm')

plt.figure(figsize=(9,9))
plt.scatter(df_usv_evt_cp['embedding_d2_0'][::nskip], 
            df_usv_evt_cp['embedding_d2_1'][::nskip],
            c = df_usv_evt_cp['main_freq_norm'][::nskip],
            s = 5,
            cmap = 'coolwarm')
plt.xlim(embedding_bw_d2.min(), embedding_bw_d2.max())
plt.ylim(embedding_bw_d2.min(), embedding_bw_d2.max())
plt.title('Main Freq on behavior UMAP')
plt.xticks()
plt.yticks()
plt.gca().set_aspect('equal', 'box')


#%%
icluster = 21
nskip = 4
df_usv_evt_cp_i = df_usv_evt_cp[df_usv_evt_cp['cluster_labels']==icluster]
plt.figure(figsize=(9,9))
plt.scatter(df_usv_evt_cp_i['embedding_d2_0'][::nskip], 
            df_usv_evt_cp_i['embedding_d2_1'][::nskip],
            s = 10,
            cmap = 'coolwarm')
plt.xlim(embedding_bw_d2.min(), embedding_bw_d2.max())
plt.ylim(embedding_bw_d2.min(), embedding_bw_d2.max())
plt.title('Main Freq on behavior UMAP')
plt.xticks()
plt.yticks()
plt.gca().set_aspect('equal', 'box')


# %%
