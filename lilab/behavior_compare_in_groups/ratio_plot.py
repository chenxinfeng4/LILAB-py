# from lilab.behavior_compare_in_groups.ratio_plot import plot_simple
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.multitest import multipletests
import pandas as pd
from scipy import stats
from lilab.comm_signal.signifiMarker import signifiMarker
from collections import OrderedDict


def merge_two_pin(df_group, column='cluster_freq', cluster_names:list=None):
    cluster_freq = np.array([*df_group[column]])
    cluster_freq [:,[9-1,20-1]] = cluster_freq[:,[9-1,20-1]].sum(axis=1, keepdims=True)
    cluster_freq [:,[23-1,34-1]] = cluster_freq[:,[23-1,34-1]].sum(axis=1, keepdims=True)
    df_group[column] = [*cluster_freq]

    if cluster_names is not None:
        cluster_names[9-1] = cluster_names[20-1] = "Pinning rotation [ 9, 20]"
        cluster_names[23-1] = cluster_names[34-1] = "Being pinned rotation [23, 34]"
    return df_group

def merge_two_chase(df_group, column='cluster_freq', cluster_names:list=None):
    cluster_freq = np.array([*df_group[column]])
    cluster_freq[:,[24-1,29-1]] = cluster_freq[:,[24-1,29-1]].sum(axis=1, keepdims=True)
    cluster_freq[:,[10-1,15-1]] = cluster_freq[:,[10-1,15-1]].sum(axis=1, keepdims=True)
    df_group[column] = [*cluster_freq]
    
    if cluster_names is not None:
        cluster_names[24-1] = cluster_names[29-1] = "Slow/fast chasing [24, 29]"
        cluster_names[10-1] = cluster_names[15-1] = "Being slow/fast chased [10, 15]"
    return df_group

def append_cluster_freq(df_group:pd.DataFrame, bhvSeqs:list, k_best:int):
    freq_l = []
    df_group = df_group.copy()
    for beh_key in df_group['beh_key']:
        seq = bhvSeqs[beh_key]
        freq_l.append([(seq==icluster+1).mean() for icluster in range(k_best)])

    df_group['cluster_freq'] = freq_l
    return df_group


def plot_simple(geno_bhv_mean, geno_bhv_sem, cluster_names, sign_p:list=None):
    #%%
    # geno_bhv_mean is size of (3, 36) and geno_bhv_sem is size of (3, 36) for 3 groups
    # Then, that's bar plot the geno_bhv_mean with error bar geno_bhv_sem and hue by group
    makers = signifiMarker(sign_p, disp_cutoff=0.1, disp_ns=False)
    plt.figure(figsize=(6,12))
    nK = geno_bhv_mean.shape[1]
    plt.barh(np.arange(nK)-0.15, geno_bhv_mean[0], xerr=geno_bhv_sem[0], height=0.3, label='Shank3HetWT', color='C1')
    plt.barh(np.arange(nK)+0.15, geno_bhv_mean[-1], xerr=geno_bhv_sem[-1], height=0.3, label='WTWT', color='gray')
    plt.yticks(np.arange(nK), cluster_names)
    plt.ylim([nK, -1])
    if makers is not None:
        for i, masker in enumerate(makers):
            plt.text(plt.xlim()[1]*0.9, i, masker, ha='right', va='center')



def stat_K36(df_group_genderA:pd.DataFrame, group_geno_l:list,
             cluster_names:list, ids_to_multitest:list = None,
             stats_method='ttest_ind'):
    geno_bhv_mean = []
    geno_bhv_sem = []
    geno_bhv = []
    for geno in group_geno_l:
        cluster_freq = df_group_genderA[df_group_genderA['group_geno']==geno]['cluster_freq']
        cluster_freq = np.array([c for c in cluster_freq], dtype=float)
        geno_bhv.append(cluster_freq)
        geno_bhv_mean.append(cluster_freq.mean(axis=0))
        geno_bhv_sem.append(cluster_freq.std(axis=0) / np.sqrt(cluster_freq.shape[0]))

    geno_bhv_mean = np.array(geno_bhv_mean)
    geno_bhv_sem = np.array(geno_bhv_sem)

    sign_p = []
    assert stats_method in ['ttest_ind', 'mannwhitneyu']
    stats_fun = lambda x,y: stats.ttest_ind(x,y, equal_var=False) if stats_method=='ttest_ind' else stats.mannwhitneyu(x,y)
    for i, (freqA, freqB) in enumerate(zip(geno_bhv[0].T, geno_bhv[-1].T)):
        # freqA = np.sort(freqA)[1:-1]
        # freqB = np.sort(freqB)[2:-2]
        if np.all(freqA==0) and np.all(freqB==0):
            p = 0.5
        else:
            p = stats_fun(freqA, freqB)[1] #mannwhitneyu, ttest_ind
        print('n1={}, n2={}, p={:.3}, {}'.format(freqA.shape[0], freqB.shape[0], p, cluster_names[i]))
        sign_p.append(p)

    sign_p = np.array(sign_p)
    # return geno_bhv_mean, geno_bhv_sem, sign_p
    if ids_to_multitest is None:
        _,sign_p,_,_=multipletests(sign_p,method='fdr_bh')
    else:
        _,sign_p[ids_to_multitest],_,_=multipletests(sign_p[ids_to_multitest],method='fdr_bh')

    return geno_bhv_mean, geno_bhv_sem, sign_p


def get_cluster_sort_ids(labels_merge_node:OrderedDict):
    """
    labels_merge_node = OrderedDict()
    labels_merge_node['Leave'] = [1,8]
    labels_merge_node['Sniff'] = [2, 4,12,26] #, 22, 36
    labels_merge_node['Chase'] = [6,19,33,24,10,29,15]
    labels_merge_node['Pounce'] = [13,27,14,28]
    labels_merge_node['Pin'] = [9,23,20,34]

    return [0, 7, 1, 3, ... , 19, 33]
    """
    cluster_sort_ids = []
    for id_list in labels_merge_node.values():
        cluster_sort_ids.extend(id_list)
    cluster_sort_ids = np.array(cluster_sort_ids) - 1 # start from 0

    n_in_each = np.array([len(id_list) for id_list in labels_merge_node.values()])
    boundary_n = n_in_each.cumsum()
    centers_n = np.array([np.concatenate([[0,], boundary_n[:-1]]), boundary_n]).mean(axis=0)

    return cluster_sort_ids, boundary_n, centers_n



def get_module_ratio(geno_bhv_mean:np.ndarray):
    geno_bhv_mean_re_treat = geno_bhv_mean[0]
    geno_bhv_mean_re_control = geno_bhv_mean[-1]
    ratio = (geno_bhv_mean_re_treat - geno_bhv_mean_re_control)/ (geno_bhv_mean_re_control+geno_bhv_mean_re_treat)
    return ratio



def plot_rawbar(geno_bhv_mean, geno_bhv_sem, sign_p, 
                labels_merge_node, cluster_names, ax=None):
    cluster_sort_ids, boundary_n, centers_n = get_cluster_sort_ids(labels_merge_node)
    geno_bhv_mean_re_treat = geno_bhv_mean[0, cluster_sort_ids]
    geno_bhv_mean_re_control = geno_bhv_mean[-1, cluster_sort_ids]
    geno_bhv_sem_re_treat = geno_bhv_sem[0, cluster_sort_ids]
    geno_bhv_sem_re_control = geno_bhv_sem[-1, cluster_sort_ids]
    cluster_names_sort = np.array(cluster_names)[cluster_sort_ids]
    nK_ = len(cluster_sort_ids)
    
    ax = plt.subplot(1,3,1) if ax is None else ax
    plt.sca(ax)
    plt.barh(np.arange(nK_)-0.15, geno_bhv_mean_re_treat, xerr=geno_bhv_sem_re_treat,
             height=0.3, label='EXP', color='C1')
    plt.barh(np.arange(nK_)+0.15, geno_bhv_mean_re_control, xerr=geno_bhv_sem_re_control,
             height=0.3, label='CON', color='gray')
    for boundary_i in boundary_n[:-1]:
        plt.axhline(y=boundary_i-0.5, color='k', linestyle='--', linewidth=1, alpha=0.5)

    for key, centers_i in zip(labels_merge_node.keys(), centers_n):
        plt.text(plt.xlim()[1]*0.95, centers_i-0.5, key, ha='right', va='center', fontsize=12)
    
    # makers = signifiMarker(sign_p, disp_cutoff=0.1, disp_ns=False)
    # for i, sig in enumerate(makers[cluster_sort_ids]):
    #     plt.text(plt.xlim()[1]*0.95, i, sig, ha='right', va='center', fontsize=12)

    plt.yticks(np.arange(nK_), cluster_names_sort)
    plt.xlabel('Percentage')
    plt.ylim([nK_, -1])
    plt.xlim([0, 0.155])
    return geno_bhv_mean_re_treat, geno_bhv_mean_re_control, cluster_names_sort



def plot_ratio(ratio, sign_p, labels_merge_node, cluster_names, ax=None):
    cluster_sort_ids, boundary_n, centers_n = get_cluster_sort_ids(labels_merge_node)
    nK_ = len(cluster_sort_ids)
    is_signifi = (sign_p<0.05)[cluster_sort_ids]

    assert len(ratio) == len(cluster_names)
    
    ratio_sort = np.array(ratio)[cluster_sort_ids]
    ax = plt.subplot(1,3,2) if ax is None else ax
    plt.barh(np.arange(nK_), ratio_sort, height=0.8, color='gray')
    plt.barh(np.arange(nK_)[is_signifi], 
            ratio_sort[is_signifi],
            height=0.8, color='C1')
    plt.xticks([-1, -0.5, 0, 0.5, 1])
    plt.yticks(np.arange(nK_), cluster_sort_ids+1)
    plt.ylim([nK_, -1])
    plt.xlim([-1, 1])
    for boundary_i in boundary_n[:-1]:
        plt.axhline(y=boundary_i-0.5, color='k', linestyle='--', linewidth=1, alpha=0.5)

    makers = signifiMarker(sign_p, disp_cutoff=0.1, disp_ns=False)
    for i, sig in enumerate(makers[cluster_sort_ids]):
        plt.text(plt.xlim()[1]*0.95, i, sig, ha='right', va='center', fontsize=12)

    plt.xlabel('CON <--> EXP')


def plot_bar_and_ratio(perc_EXP_eV:np.ndarray, perc_CON_eV:np.ndarray, sign_p:np.ndarray, cluster_names:list):
    ntrial, nK_ = perc_EXP_eV.shape
    assert perc_CON_eV.shape[1] == len(sign_p) == len(cluster_names) == nK_
    is_signifi = (sign_p<0.05)

    plt.figure(figsize=(8, 4))
    plt.subplot(1,3,2)

    perc_EXP_mean = np.mean(perc_EXP_eV, axis=0)
    perc_CON_mean = np.mean(perc_CON_eV, axis=0)
    perc_EXP_sem = np.std(perc_EXP_eV, axis=0) / np.sqrt(perc_EXP_eV.shape[0] - 1)
    perc_CON_sem = np.std(perc_CON_eV, axis=0) / np.sqrt(perc_CON_eV.shape[0] - 1)
    plt.barh(np.arange(nK_)-0.2, perc_EXP_mean, height=0.4, color='C1', label='EXP')
    plt.barh(np.arange(nK_)+0.2, perc_CON_mean, height=0.4, color='gray', label='CON')

    plt.errorbar(perc_EXP_mean, np.arange(nK_)-0.2, xerr=perc_EXP_sem, fmt='none', capsize=3, ecolor='k')
    plt.errorbar(perc_CON_mean, np.arange(nK_)+0.2, xerr=perc_CON_sem, fmt='none', capsize=3, ecolor='k')
    plt.ylim(nK_ - 0.5, -0.5)
    plt.legend()
    makers = signifiMarker(sign_p, disp_cutoff=0.1, disp_ns=False)
    for i, sig in enumerate(makers):
        plt.text(plt.xlim()[1]*0.95, i, sig, ha='right', va='center', fontsize=12)
    plt.yticks(np.arange(nK_), cluster_names)
    plt.xlabel('Percentage')

    ratio = (perc_EXP_mean - perc_CON_mean) / (perc_EXP_mean + perc_CON_mean + 0.001)
    ratio_each = (perc_EXP_eV - perc_CON_mean[None]) / (perc_EXP_eV + perc_CON_mean[None] + 0.001)
    ratio_m = np.mean(ratio_each, axis=0)
    ratio_sem = np.std(ratio_each, axis=0) / np.sqrt(perc_EXP_eV.shape[0] - 1)
    
    plt.subplot(1,3,3)
    plt.barh(np.arange(nK_), ratio, height=0.8, color='gray')
    plt.barh(np.arange(nK_)[is_signifi], 
            ratio[is_signifi],
            height=0.8, color='C1')
    plt.axvline(0, color='k', linestyle='-', linewidth=0.75)
    plt.xticks([-1, -0.5, 0, 0.5, 1])
    plt.yticks(np.arange(nK_), np.arange(nK_)+1)
    plt.ylim(nK_ - 0.5, -0.5)
    plt.xlim([-1, 1])
    plt.errorbar(ratio, np.arange(nK_), xerr=ratio_sem, fmt='none', capsize=3, ecolor='k')
    makers = signifiMarker(sign_p, disp_cutoff=0.1, disp_ns=False)
    for i, sig in enumerate(makers):
        plt.text(plt.xlim()[1]*0.95, i, sig, ha='right', va='center', fontsize=12)
    plt.xlabel('CON <--> EXP')
