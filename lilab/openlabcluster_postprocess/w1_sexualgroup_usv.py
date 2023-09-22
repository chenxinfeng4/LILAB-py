# %%
import os
import pickle as pkl
import numpy as np
import os.path as osp
import matplotlib
#matplotlib.use('agg') 
import matplotlib.pyplot as plt
from lilab.openlabcluster_postprocess.s1_merge_3_file import get_assert_1_file
import pandas as pd
from statannotations.Annotator import Annotator
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from numpy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from lilab.openlabcluster_postprocess.u1_percent_rat_boxplot_CompPairPro import (
    define_group,  create_nodemerge_group_x_freq,get_behavior_percentages,
    get_clean_df, plot_box_data, plot_dim_reduced_2d, plot_dim_reduced_3d
)
from lilab.openlabcluster_postprocess.u1_percent_rat_boxplot_CompAnova import (
    plt_box_anova
)

project = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/DayAll20230828/'

sheet_name = '熊组合作D75_treat_info'

def load_data(project, sheet_name):
    clippredpkl = get_assert_1_file(osp.join(project,'usv_cluster/*.usvclippredpkl'))
    clippredata = pkl.load(open(clippredpkl,'rb'))
    cluster_labels = clippredata['cluster_labels']
    clipNames = clippredata['clipNames']

    k_best = clippredata['ncluster'] #label: 0,1,2...,k_best-1
    clippredata['cluster_names'] = [f'{i}' for i in range(k_best)]
    lab_names2 = [f'{lab} [{i:>2}]' for i, lab in enumerate(clippredata['cluster_names'])]
    df_labnames = pd.DataFrame({'behlabel':range(k_best), 'lab_names':lab_names2})
 
    # read group information
    groupFile=get_assert_1_file(osp.join(project,'rat_info/*rat_*info*.xlsx'))
    rat_info=pd.read_excel(groupFile, sheet_name='rat_info', engine='openpyxl')
    video_info=pd.read_excel(groupFile, sheet_name=sheet_name, engine='openpyxl')

    rat_info = rat_info.filter(regex=r'^((?!Unnamed).)*$')
    video_info = video_info.filter(regex=r'^((?!Unnamed).)*$')

    # 表的检查
    assert {'animal', 'color', 'gender', 'dob'} <= set(rat_info.columns)
    assert {'video_nake', 'animal', 'partner', 'usv_file'} <= set(video_info.columns)
    df_merge_b = pd.merge(rat_info, video_info['animal'], on='animal', how='right')
    assert (df_merge_b['color'] == 'b').all()
    df_merge_w = pd.merge(rat_info, video_info['partner'], left_on='animal', right_on='partner', how='right')
    assert (df_merge_w['color'] == 'w').all()
    return rat_info, video_info, df_labnames, k_best, cluster_labels, clipNames

def get_behavior_count(bhv, bhv_unique):
    return [np.sum(bhv==c) for c in bhv_unique] 

def define_group_freq(df_clipNames, df_group1, k_best):
    freq_data = []
    for video_nake in df_group1['video_nake'].values:
        bhv = df_clipNames[df_clipNames['video_nake']==video_nake]['cluster_labels'].values
        freq_data_ =  get_behavior_count(bhv, np.arange(k_best))
        freq_data.append(freq_data_)
    freq_data = np.array(freq_data)
    df_freq_data_list = []
    for i, freq_data_this in enumerate(freq_data):
        df_freq_data = pd.DataFrame({'behlabel':range(len(freq_data_this)), 'freq':freq_data_this})
        df_freq_data['video_nake'] = df_group1['video_nake'].values[i]
        df_freq_data_list.append(df_freq_data)
    df_freq_data_list = pd.concat(df_freq_data_list, axis=0)
    df_group_x_freq = pd.merge(pd.merge(df_group1, df_freq_data_list, on='video_nake'), df_labnames, on='behlabel')
    return freq_data, df_group_x_freq

# %%
rat_info, video_info, df_labnames, k_best, cluster_labels, clipNames = load_data(project, sheet_name)
df_clipNames = pd.DataFrame(clipNames, columns=['video_nake', 'ind'])
df_clipNames['cluster_labels'] = cluster_labels
df_group = define_group(rat_info, video_info, df_labnames)

df_group1 = df_group[df_group['is_blackfirst']==True]
df_group1.loc[df_group1['group']=='femalemale', 'group'] = 'malefemale'
df_group1 = df_group1.drop(['is_blackfirst', 'beh_key'], axis=1)

freq_data, df_group_x_freq = define_group_freq(df_clipNames, df_group1, k_best)

hue_order = ['malemale', 'femalefemale']
hue_legend = ['M_M', 'F_F']
freq_data = freq_data[~np.isnan(freq_data)]
df_group_x_freq = df_group_x_freq.dropna()
plot_box_data(df_labnames, df_group_x_freq, hue_order, hue_legend)
plt.xlabel('Count (in 15min)')


hue_order = ['malemale', 'malefemale', 'femalefemale']
hue_legend = ['M_M', 'M_F, F_M', 'F_F']
hue_color = ['#313695', '#74add1', '#a50026']
plt_box_anova(df_labnames, df_group_x_freq, hue_order, hue_legend, hue_color)
plt.xlabel('Count (in 15min)')
plt.title(sheet_name)
plt.savefig(osp.join(project, 'usv_cluster', 'boxplot_MM_vs_FF_D75_USV.pdf'))
# %%
