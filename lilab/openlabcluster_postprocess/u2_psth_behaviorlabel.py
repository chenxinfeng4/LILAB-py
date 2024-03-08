# %%
import numpy as np
import matplotlib.pyplot as plt
from lilab.openlabcluster_postprocess.u1_percent_rat_boxplot_CompPairPro import (
    load_data, define_group
)
from sklearn.preprocessing import OneHotEncoder
from lilab.comm_signal.detectTTL import detectTTL
from lilab.comm_signal.BF_AlignWave2Tg import BF_AlignWave2Tg


project='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day75'
sheet_name = '熊组合作D75_treat_info'
rat_info, video_info, bhvSeqs, df_labnames, k_best, cluster_nodes_merged = load_data(project, sheet_name)
df_group = define_group(rat_info, video_info, df_labnames)
# %%
groupA = 'malemale'
df_groupA = df_group[df_group['group'] == groupA]

def concat_lab_seq(df_groupA):
    lab_seq = np.concatenate([bhvSeqs[beh_key] for beh_key in df_groupA['beh_key']])
    #downsample from 30Hz to 5Hz
    lab_seq = lab_seq[::6]
    fs = 5
    # convert lab_seq to one hot encoding array
    encoder = OneHotEncoder()
    one_hot_lab_seq = encoder.fit_transform(lab_seq.reshape(-1, 1)).toarray()
    tRise_l = [detectTTL(one_hot_lab_seq[:,i], 'up-up', 10, fs=fs)[0]
               for i in range(one_hot_lab_seq.shape[1])]
    
    return fs, one_hot_lab_seq, tRise_l

fs, one_hot_lab_seq, tRise_l = concat_lab_seq(df_groupA)

twin = [-10, 10]
one_hot_lab_seq_freq = one_hot_lab_seq.mean(axis=0)

#%%
tg_evt = 14
for wave_evt in range(10, 39):
# for wave_evt in [23,26,17]:
# for wave_evt in [32,9]:
    # wave_evt = 20
    if tg_evt==wave_evt: continue
    algned_wave = BF_AlignWave2Tg(one_hot_lab_seq[:,wave_evt], tRise_l[tg_evt], *twin, fs)
    algned_wave_meam = algned_wave.mean(axis=0) / one_hot_lab_seq_freq[wave_evt]
    # if not np.sum(algned_wave_meam>4)>3: continue
    xticks = np.linspace(*twin, len(algned_wave_meam))
    plt.plot(xticks, algned_wave_meam, label=df_labnames.iloc[wave_evt]['lab_names'])

    # algned_wave_std  = algned_wave.std(axis=-1)
    # algned_wave_sem = algned_wave_std/np.sqrt(algned_wave.shape[1])
    # BF_plotwSEM(xticks, algned_wave_meam, algned_wave_sem)
plt.legend(bbox_to_anchor=(1, 0.5))
plt.xlabel(df_labnames.iloc[tg_evt]['lab_names'])
plt.xticks([-10,-5,0,5,10])
# plt.axvline(x=0.5, color='k', linestyle='-')
# plt.axhline(y=4, color='k', linestyle='-')
plt.ylabel('Relative Frequence')
# %%
