# %%
import numpy as np
import pandas as pd
import os.path as osp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from lilab.openlabcluster_postprocess.u1_percent_rat_boxplot_CompPairPro import (
    load_data, define_group, define_group_freq, 
    create_nodemerge_group_x_freq, get_clean_df
)
from sklearn.preprocessing import OneHotEncoder
from lilab.comm_signal.detectTTL import detectTTL
from lilab.comm_signal.cutwave import cutwave
from lilab.comm_signal.BF_AlignSg2Tg import BF_AlignSg2Tg
from lilab.comm_signal.BF_AlignWave2Tg import BF_AlignWave2Tg
from lilab.comm_signal.BF_plotwSEM import BF_plotwSEM

project='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day75'
sheet_name = '熊组合作D75_treat_info'
rat_info, video_info, bhvSeqs, df_labnames, k_best, cluster_nodes_merged = load_data(project, sheet_name)
df_group = define_group(rat_info, video_info, df_labnames)
# %%
groupA = 'malemale'
df_groupA = df_group[df_group['group'] == groupA]

lab_seq = np.concatenate([bhvSeqs[beh_key] for beh_key in df_groupA['beh_key']])
lab_seq_squeeze = lab_seq[1:][np.diff(lab_seq)!=0]

tg_evt = 14
tg_evt_tick = lab_seq_squeeze==tg_evt
tg_evt_tick[:10] = False
tg_evt_tick[-10:] = False

tg_evt_tick_t = np.where(tg_evt_tick)[0]
tg_evt_tick_rg = tg_evt_tick_t[:,None] + [-5, 0]
sig_evt_list  = []
for tg_evt_tick_rg_i in tg_evt_tick_rg:
    sig_evt_list.append(lab_seq_squeeze[tg_evt_tick_rg_i[0] : tg_evt_tick_rg_i[1]])
sig_evt = np.stack(sig_evt_list)

# %%
unique_values, counts = np.unique(sig_evt.flatten(), return_counts=True)