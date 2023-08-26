# %%
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from lilab.openlabcluster_postprocess.u1_percent_rat_boxplot_CompPairPro import (
    load_data, define_group
)
from lilab.openlabcluster_postprocess.s1_merge_3_file import get_assert_1_file
from lilab.comm_signal.detectTTL import detectTTL
from lilab.comm_signal.BF_AlignSg2Tg import BF_AlignSg2Tg
from lilab.comm_signal.BF_plotRaster import BF_plotRasterCell
import pickle
from collections import defaultdict


project='/mnt/liying.cibr.ac.cn_Xiong/USV_MP4-toXiongweiGroup/Shank3_USV/'
sheet_name = 'treat_info'
rat_info, video_info, bhvSeqs, df_labnames, k_best, cluster_nodes_merged = load_data(project, sheet_name)
df_group = define_group(rat_info, video_info, df_labnames)

usv_file = get_assert_1_file(osp.join(project, 'usv_label', '*.usvpkl'))
usv_file_data = pickle.load(open(usv_file, 'rb'))


# %%
video_nakes = set(usv_file_data['video_nakes'])
df_group_usv = df_group[df_group['video_nake'].isin(video_nakes)]
df_tick_usv = usv_file_data['df_usv_evt']
# %%
# 1个 usv 文件对应 2个 beh_key
usv_evt_dict = defaultdict(list)
twin = [-10, 10]
fs = 30
behlabel_unique = df_labnames['behlabel'].values
for i in range(len(df_group_usv)):
    beh_key = df_group_usv.iloc[i]['beh_key']
    video_nake = df_group_usv.iloc[i]['video_nake']
    bhvSeq = np.array(bhvSeqs[beh_key])
    usv_start = df_tick_usv[df_tick_usv['video_nake'] == video_nake]['start'].values
    for behlabel in behlabel_unique:
        if behlabel not in bhvSeq: continue
        trigger = detectTTL(bhvSeq==behlabel, 'up-up', 4, fs=fs)[0]
        Alg_cell = BF_AlignSg2Tg(usv_start, trigger, *twin)
        usv_evt_dict[behlabel].extend(Alg_cell)
        
# %%

for i in range(len(df_group_usv)):
    if i%4 ==0:
        fig,ax = plt.subplots(2,2, figsize=(10,10))
        ax = ax.flatten()
    usv_evt_now = usv_evt_dict[i]
    if usv_evt_now == []: continue
    ax_now = ax[i%4]
    BF_plotRasterCell(usv_evt_now, ax=ax_now)
    ax_now.set_xlim([-10,10])
    ax_now.set_xticks([-10,-5, 0,5,10])
    ax_now.set_ylim([0, len(usv_evt_now)])
    ax_now.set_title(df_labnames['lab_names'].iloc[i])
