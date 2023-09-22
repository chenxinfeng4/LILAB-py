#%%
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
    load_data, define_group, define_group_freq, create_nodemerge_group_x_freq,
    get_clean_df, plot_box_data, plot_dim_reduced_2d, plot_dim_reduced_3d
)

project='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/DayAll20230828'

sheet_name_list = ['熊组合作D35_treat_info', '熊组合作D55_treat_info', '熊组合作D75_treat_info']
sheet_group_legend = ['D35', 'D55', 'D75']

def load_data_from_sheet_group(project, sheet_group_legend, sheet_name_list):
    bhvSeqs = dict()
    video_info = pd.DataFrame()
    for sheet_group, sheet_name in zip(sheet_group_legend, sheet_name_list):
        rat_info, video_info_, bhvSeqs_, df_labnames, k_best, cluster_nodes_merged = load_data(project, sheet_name)
        video_info_['sheet_group'] = sheet_group
        video_info = video_info.append(video_info_)
        bhvSeqs.update(bhvSeqs_)
    video_info.reset_index(inplace=True)
    return rat_info, video_info, bhvSeqs, df_labnames, k_best, cluster_nodes_merged

# %%
def define_group_from_sheet_group(rat_info, video_info, sheet_group_legend):
    df_group = pd.DataFrame()
    for sheet_group in sheet_group_legend:
        video_info_ = video_info[video_info['sheet_group'] == sheet_group]
        df_group_ = define_group(rat_info, video_info_, None)
        df_group_['sheet_group'] = sheet_group
        df_group = df_group.append(df_group_)
    df_group.reset_index(inplace=True)
    return df_group


def get__df_group_x_freq(rat_info, video_info, bhvSeqs, df_labnames, cluster_nodes_merged):
    df_group = define_group_from_sheet_group(rat_info, video_info, sheet_group_legend)
    df_group['sex_group'] = df_group['group']
    df_group['group'] = df_group['sheet_group'] + '_' + df_group['sex_group']

    freq_data, df_group_x_freq = define_group_freq(bhvSeqs, df_group, df_labnames)
    df_group_x_freq_nodemerge = create_nodemerge_group_x_freq(df_group_x_freq, cluster_nodes_merged)


    df_group_x_freq_cp = df_group_x_freq.copy()
    df_group_x_freq_nodemerge = df_group_x_freq_nodemerge.copy()
    df_group_x_freq = get_clean_df(df_group_x_freq)
    df_group_x_freq_nodemerge = get_clean_df(df_group_x_freq_nodemerge)
    return df_group, df_group_x_freq, df_group_x_freq_nodemerge, freq_data


if __name__ == "__main__":
    rat_info, video_info, bhvSeqs, df_labnames, k_best, cluster_nodes_merged = load_data_from_sheet_group(project, sheet_group_legend, sheet_name_list)
    df_group, df_group_x_freq, df_group_x_freq_nodemerge, freq_data = get__df_group_x_freq(rat_info, video_info, bhvSeqs, df_labnames, cluster_nodes_merged)

    for age in sheet_group_legend:
        hue_order = [f'{age}_malemale', f'{age}_femalefemale']
        hue_legend = [f'M_M({age})', f'F_F({age})']
        plot_box_data(df_labnames, df_group_x_freq, hue_order, hue_legend, 
                    savefigpath=osp.join(project, f'boxplot_MM_vs_FF_{age}.pdf'))
        plot_box_data(df_labnames, df_group_x_freq_nodemerge, hue_order, hue_legend,
                    savefigpath=osp.join(project, f'boxplot_MM_vs_FF_nodemerge_{age}.pdf'))

        hue_order = [f'{age}_malefemale', f'{age}_femalemale']
        hue_legend = [f'[M]_F({age})', f'[F]_M({age})']
        plot_box_data(df_labnames, df_group_x_freq, hue_order, hue_legend,
                    savefigpath=osp.join(project, f'boxplot_MF_vs_FM_{age}.pdf'))
        plot_box_data(df_labnames, df_group_x_freq_nodemerge, hue_order, hue_legend,
                    savefigpath=osp.join(project, f'boxplot_MF_vs_FM_nodemerge_{age}.pdf'))

    # %%
    data_dict = {'group': df_group['group'].values}
    data_df = pd.DataFrame(data_dict)

    list_geno = ['malemale', 'femalefemale', 'malefemale', 'femalemale']
    legend_list_geno = ['M_M', 'F_F', '<M>_F', '<F>_M']
    colors = ['#313695', '#a50026', '#74add1', '#f46d43']
    data_pca = PCA(n_components=3).fit_transform(
                    freq_data[:,1:])  #not include nonsocial

    data_lda = LinearDiscriminantAnalysis(n_components=3).fit_transform(
                    PCA(n_components=8).fit_transform(freq_data[:,1:]), 
                    df_group['group'].values)     #not include nonsocial


    list_geno = ['D35_malemale', 'D35_femalefemale', 
                'D55_malemale', 'D55_femalefemale',
                'D75_malemale', 'D75_femalefemale']
    legend_list_geno = ['M_M(D35)', 'F_F(D35)',
                        'M_M(D55)', 'F_F(D55)',
                        'M_M(D75)', 'F_F(D75)']
    colors = ['#313695', '#a50026', '#74add1', '#f46d43', '#abd9e9', '#fdae61']
    # colors = ['#313695', '#a50026', '#74add1', '#f46d43']
    data_pca = PCA(n_components=3).fit_transform(
                    freq_data[:,1:])  #not include nonsocial

    data_lda = LinearDiscriminantAnalysis(n_components=3).fit_transform(
                    PCA(n_components=8).fit_transform(freq_data[:,1:]), 
                    df_group['group'].values)     #not include nonsocial

    data_dict = {'group': df_group['group'].values}
    data_df = pd.DataFrame(data_dict)

    fig = plt.figure(figsize=(14, 14))
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224)

    plot_dim_reduced_2d(data_pca, data_df, 'PC', list_geno, legend_list_geno, colors, ax=ax2, savefigpath=None)
    plot_dim_reduced_2d(data_lda, data_df, 'LD', list_geno, legend_list_geno, colors, ax=ax4, savefigpath=None)

    plot_dim_reduced_3d(data_pca, data_df, 'PC', list_geno, legend_list_geno, colors, ax=ax1, savefigpath=None)
    plot_dim_reduced_3d(data_lda, data_df, 'LD', list_geno, legend_list_geno, colors, ax=ax3, savefigpath=None)

    plt.savefig(osp.join(project, 'PCA_LDA.pdf'), bbox_inches='tight')
