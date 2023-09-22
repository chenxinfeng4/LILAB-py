# %%
import pickle
import numpy as np
from lilab.openlabcluster_postprocess.u1_percent_rat_boxplot_CompPairPro import (
    load_data, define_group
)
import os.path as osp
import matplotlib.pyplot as plt
from lilab.openlabcluster_postprocess.s1_merge_3_file import get_assert_1_file
from lilab.feature_autoenc.heatmap_functions import HeatmapType, heatmap


# %%
project='/mnt/liying.cibr.ac.cn_Xiong/USV_MP4-toXiongweiGroup/Shank3_USV/'
sheet_name = 'treat_info'
rat_info, video_info, bhvSeqs, df_labnames, k_best, cluster_nodes_merged = load_data(project, sheet_name)
df_group = define_group(rat_info, video_info, df_labnames)

usv_evt_file = get_assert_1_file(osp.join(project, 'usv_label', '*_evt.usvpkl'))
usv_evt_data = pickle.load(open(usv_evt_file, 'rb'))


usv_latent_file = get_assert_1_file(osp.join(project, 'usv_label', '*_latent.usvpkl'))
usv_latent_data = pickle.load(open(usv_latent_file, 'rb'))
usv_latent_dict = usv_latent_data['usv_latent']

video_nakes = set(usv_evt_data['video_nakes'])
df_group_usv = df_group[df_group['video_nake'].isin(video_nakes)]
df_tick_usv = usv_evt_data['df_usv_evt']

# %%
records = []
fps = 30
usv_latent_tsne = usv_latent_data['usv_latent_tsne']
for video_nake in usv_latent_data['video_nakes']:
    rat_wF = f'fps30_{video_nake}_startFrame0_whiteFirst'.replace('-','_')
    rat_bF = f'fps30_{video_nake}_startFrame0_blackFirst'.replace('-','_')
    label_wF = np.array(bhvSeqs[rat_wF])
    label_bF = np.array(bhvSeqs[rat_bF])

    ind_tick = df_tick_usv[df_tick_usv['video_nake']==video_nake]['start'].values * fps
    ind_tick = np.clip(ind_tick, 0, 27000-1).astype(int)

    label_usv_tick_wF = label_wF[ind_tick]
    label_usv_tick_bF = label_bF[ind_tick]
    record = np.stack([label_usv_tick_wF, label_usv_tick_bF], axis=-1)
    records.append(record)
records = np.concatenate(records, axis=0)

# %%
usv_latent_tsne_dup = np.concatenate([usv_latent_tsne, usv_latent_tsne], axis=0)
records_dup = np.concatenate([records[:,0], records[:,1]], axis=0)

for ilabel in range(k_best):
    usv_latent_tsne_ilabel = usv_latent_tsne_dup[records_dup==ilabel]
    nsample = len(usv_latent_tsne_ilabel)
    if nsample <= 400: continue
    heat_tsne2d,xedges,yedges = heatmap(usv_latent_tsne_ilabel, bins=80)
    heatobj = HeatmapType(heat_tsne2d, xedges, yedges)
    heatobj.calculate()
    heatobj.plt_heatmap()
    plt.xticks([-10,])
    plt.yticks([-10,])
    plt.xlabel('t-SNE1', fontsize=16)
    plt.ylabel('t-SNE2', fontsize=16)
    plt.title(f't-SNE of {ilabel}, n={nsample}', fontsize=16)
    plt.show()
    plt.close()

node_merge = [4, 7, 43, 38, 17] #pin
node_merge = [27, 28, 40] # rear
node_merge = [33, 6, 1] 
node_merge = [11, 13, 32]  # approach, chase
#%%
node_merge = [29, 22, 42]  # approach, chase

usv_latent_tsne_ilabel = usv_latent_tsne_dup[np.isin(records_dup, node_merge)]
nsample = len(usv_latent_tsne_ilabel)
heat_tsne2d,xedges,yedges = heatmap(usv_latent_tsne_ilabel, bins=80)
heatobj = HeatmapType(heat_tsne2d, xedges, yedges)
heatobj.calculate()
heatobj.plt_heatmap()
plt.xticks([-10,])
plt.yticks([-10,])
plt.xlabel('t-SNE1', fontsize=16)
plt.ylabel('t-SNE2', fontsize=16)
plt.title(f't-SNE of {node_merge}, n={nsample}', fontsize=16)
plt.show()
plt.close()

# %%
heat_tsne2d,xedges,yedges = heatmap(usv_latent_tsne_dup, bins=80)
heatobj = HeatmapType(heat_tsne2d, xedges, yedges)
heatobj.calculate()
heatobj.plt_heatmap()
plt.xticks([-10,])
plt.yticks([-10,])
plt.xlabel('t-SNE1', fontsize=16)
plt.ylabel('t-SNE2', fontsize=16)
plt.title(f't-SNE of {node_merge}, n={nsample}', fontsize=16)
plt.show()
plt.close()

# %%
heat_tsne2d= usv_latent_data['usv_latent_tsne']
plt.plot(heat_tsne2d[:,0], heat_tsne2d[:,1], 'k.', markersize=1)
plt.axis('off')
# set axis equal
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
plt.close()
# %%
