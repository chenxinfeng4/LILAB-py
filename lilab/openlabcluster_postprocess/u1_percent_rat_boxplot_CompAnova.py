#%%
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


def plt_box_anova(df_labnames, df_group_x_freq, hue_order, hue_legend, hue_color, 
                  ignore_class0=False, savefigpath=None):
    assert len(hue_order) > 2
    hue_ref = -1
    hue_subs = hue_order[:-1]
 
    if ignore_class0:
        df_group_x_freq = df_group_x_freq[df_group_x_freq['behlabel']!=0] #ignore non social

    df_group_x_freq_ref = df_group_x_freq[df_group_x_freq['group']==hue_order[-1]]
    behlabel_unique = df_group_x_freq['behlabel'].unique()
    lab_names_unique = df_group_x_freq['lab_names'].unique()
    fold_changes = []
    for i in behlabel_unique:
        df_this_hue = [df_group_x_freq[df_group_x_freq['group']==hue_sub] for hue_sub in hue_subs]
        df_this_hue_thislabel = [df_this_hue_[df_this_hue_['behlabel']==i]['freq'].mean()
                                    for df_this_hue_ in df_this_hue]
        df_ref_hue_thislabel = df_group_x_freq_ref[df_group_x_freq_ref['behlabel']==i]['freq'].mean()
        fold_changes.append(np.max(df_this_hue_thislabel) / df_ref_hue_thislabel)
    fold_changes = np.array(fold_changes)
    fc_inds=np.argsort(fold_changes)[::-1]
    behlabel_sort = behlabel_unique[fc_inds]
    lab_names_sort = lab_names_unique[fc_inds]

    xcolors = []
    p_values = []
    for i in behlabel_unique:
        df_group_x_freq_i = df_group_x_freq[df_group_x_freq['behlabel']==i]
        groups = [df_group_x_freq_i[df_group_x_freq_i["group"]==hue]['freq'].values
                    for hue in hue_order]
        f_value, p_value = stats.f_oneway(*groups)
        p_values.append(p_value)
        if p_value >= 0.05:
            xcolor = 'black'
        else:
            ind = np.argmax([np.mean(g) for g in groups])
            xcolor = hue_color[ind]
        xcolors.append(xcolor)
    xcolors_sort = np.array(xcolors)[fc_inds]
    p_values_sort = np.array(p_values)[fc_inds]

    palette=dict(zip(hue_order, hue_color))
    # plt.figure(figsize = (15,25))
    plt.figure(figsize = (10,15))
    sns.boxplot(y='lab_names', x='freq', hue='group', 
                hue_order=hue_order,
                data=df_group_x_freq,
                width=0.75,
                order=lab_names_sort,
                palette=palette,
                orient='h')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Percentage', fontsize=20)
    plt.ylabel('Label', fontsize=20)
    leg = plt.legend(fontsize=14)
    for i, text in enumerate(hue_legend): leg.get_texts()[i].set_text(text)

    ax = plt.gca()
    for kbi in range(len(behlabel_unique)):
       ax.get_yticklabels()[kbi].set_color(xcolors_sort[kbi])

    if savefigpath:
        plt.savefig(savefigpath, bbox_inches='tight')

#%%
if __name__ == '__main__':
    project='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day75'
    sheet_name = '熊组合作D75_treat_info'
    rat_info, video_info, bhvSeqs, df_labnames, k_best, cluster_nodes_merged = load_data(project, sheet_name)
    df_group = define_group(rat_info, video_info, df_labnames)
    freq_data, df_group_x_freq = define_group_freq(bhvSeqs, df_group, df_labnames)
    df_group_x_freq_nodemerge = create_nodemerge_group_x_freq(df_group_x_freq, cluster_nodes_merged)

    df_group_x_freq_cp = df_group_x_freq.copy()
    df_group_x_freq_nodemerge = df_group_x_freq_nodemerge.copy()
    df_group_x_freq = get_clean_df(df_group_x_freq)
    df_group_x_freq_nodemerge = get_clean_df(df_group_x_freq_nodemerge)

    if False:
        hue_order = ['malemale', 'malefemale', 'femalemale', 'femalefemale']
        hue_legend = ['<M>_M', '<M>_F', '<F>_M', '<F>_F']
        hue_color = ['#313695', '#74add1', '#f46d43','#a50026']
        plt_box_anova(df_labnames, df_group_x_freq, hue_order, hue_legend, hue_color)
        plt_box_anova(df_labnames, df_group_x_freq_nodemerge, hue_order, hue_legend, hue_color)

        plt_box_anova(df_labnames, df_group_x_freq, hue_order, hue_legend, 
                    savefigpath=osp.join(project, 'boxplot_MM_MF_FM_FF.pdf'))
        plt_box_anova(df_labnames, df_group_x_freq_nodemerge, hue_order, hue_legend,
                    savefigpath=osp.join(project, 'boxplot_MM_MF_FM_FF_nodemerge.pdf'))

    if True:
        df_group_x_freq_nodemerge_groupmerge = df_group_x_freq_nodemerge.copy()
        df_group_x_freq_nodemerge_groupmerge[df_group_x_freq_nodemerge_groupmerge['group']=='femalemale']['group'] = 'malefemale'
        hue_order = ['malemale', 'malefemale', 'femalefemale']
        hue_legend = ['M_M', 'M_F, F_M', 'F_F']
        hue_color = ['#313695', '#74add1', '#a50026']
        plt_box_anova(df_labnames, df_group_x_freq_nodemerge_groupmerge, hue_order, hue_legend, hue_color)

        df_group_x_freq_groupmerge = df_group_x_freq.copy()
        df_group_x_freq_groupmerge[df_group_x_freq_groupmerge['group']=='femalemale']['group'] = 'malefemale'
        hue_order = ['malemale', 'malefemale', 'femalefemale']
        hue_legend = ['M_M', 'M_F, F_M', 'F_F']
        hue_color = ['#313695', '#74add1', '#a50026']
        plt_box_anova(df_labnames, df_group_x_freq_groupmerge, hue_order, hue_legend, hue_color)

    if True:
        lab_name_unique = df_group_x_freq_nodemerge_groupmerge['lab_names'].unique()
        outarray = np.zeros((len(lab_name_unique), len(hue_order)))
        for ihue, hue in enumerate(hue_order):
            for ilab, lab_name in enumerate(lab_name_unique):
                outarray[ilab, ihue] = df_group_x_freq_nodemerge_groupmerge[(df_group_x_freq_nodemerge_groupmerge['lab_names'] == lab_name) &
                                                    (df_group_x_freq_nodemerge_groupmerge['group'] == hue) ]['freq'].mean()

        df_ternary = pd.DataFrame(outarray, columns=hue_legend)
        df_ternary['lab_names'] = lab_name_unique
        import plotly.express as px
        fig = px.scatter_ternary(df_ternary, a=hue_legend[0], b=hue_legend[1], c=hue_legend[2], hover_name="lab_names")
        fig.show()
