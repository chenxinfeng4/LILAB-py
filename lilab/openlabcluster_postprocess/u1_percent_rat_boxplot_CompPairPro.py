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
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from numpy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.stats.multitest import multipletests
#%%
##path
project='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day35'
sheet_name = '熊组合作D35_treat_info'

# project='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day55'
# sheet_name = '熊组合作D55_treat_info'

# project='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day75'
# sheet_name = '熊组合作D75_treat_info'

def load_data(project, sheet_name):
    bhv_seqFile = get_assert_1_file(osp.join(project,'*_sequences.pkl'))
    bhvSeqs = pkl.load(open(bhv_seqFile,'rb'))
    #print(list(bhvSeqs.keys()))  #label: 0,1,2...,k_best

    clippredpkl = get_assert_1_file(osp.join(project,'*.clippredpkl'))
    clippredata = pkl.load(open(clippredpkl,'rb'))
    cluster_nodes_merged = clippredata.get('cluster_nodes_merged', [])

    k_best = clippredata['ncluster'] #label: 0,1,2...,k_best
    assert len(clippredata['cluster_names']) == k_best  #不包含nonsocial
    lab_names2 = [f'{lab} [{i:>2}]' for i, lab in enumerate(['Far away non social']+clippredata['cluster_names'])]
    df_labnames = pd.DataFrame({'behlabel':range(k_best+1), 'lab_names':lab_names2})
    df_labreorder = pd.DataFrame({'behlabel':range(k_best+1)})
 
    # read group information
    groupFile=get_assert_1_file(osp.join(project,'rat_info/*rat_*info*.xlsx'))
    rat_info=pd.read_excel(groupFile, sheet_name='rat_info', engine='openpyxl')
    video_info=pd.read_excel(groupFile, sheet_name=sheet_name, engine='openpyxl')

    rat_info = rat_info.filter(regex=r'^((?!Unnamed).)*$').dropna(axis=0,how='all')
    video_info = video_info.filter(regex=r'^((?!Unnamed).)*$').dropna(axis=0,how='all')
    #remove all NaN row
    # 表的检查
    assert {'animal', 'color', 'gender', 'geno', 'dob'} <= set(rat_info.columns)
    assert {'video_nake', 'animal', 'partner'} <= set(video_info.columns)
    df_merge_b = pd.merge(rat_info, video_info['animal'], on='animal', how='right')
    print(df_merge_b['color'].unique())
    assert (df_merge_b['color'] == 'b').all()
    df_merge_w = pd.merge(rat_info, video_info['partner'], left_on='animal', right_on='partner', how='right')
    print(df_merge_w['color'].unique())
    assert (df_merge_w['color'] == 'w').all()
    return rat_info, video_info, bhvSeqs, df_labnames, k_best, cluster_nodes_merged


def define_group(rat_info, video_info, df_labnames=None):
    rats_black = rat_info[rat_info['color'] == 'b']['animal'].values
    rats_male  = rat_info[rat_info['gender'] == 'male']['animal'].values
    group_geno_dict = dict(rat_info[['animal', 'geno']].values)
    rows_list = []
    for i in range(video_info.shape[0]):
        record_now = video_info.iloc[i]
        video_nake = record_now['video_nake']

        animal_ratid, partner_ratid = record_now['animal'], record_now['partner']
        is_blackfirst = animal_ratid in rats_black
        black_ratid, white_ratid = (animal_ratid, partner_ratid) if is_blackfirst else (partner_ratid, animal_ratid) #first_ratid is black_rat
        groupdict = {True:'male', False:'female'}
        group_name = groupdict[black_ratid in rats_male] + groupdict[white_ratid in rats_male]
        group_geno_name = group_geno_dict[black_ratid] + group_geno_dict[white_ratid]
        rows_list.append([black_ratid, white_ratid, group_name, group_geno_name, video_nake, True])

        group_name = groupdict[white_ratid in rats_male] + groupdict[black_ratid in rats_male]
        group_geno_name = group_geno_dict[white_ratid] + group_geno_dict[black_ratid]
        rows_list.append([white_ratid, black_ratid, group_name, group_geno_name, video_nake, False])

    df_group = pd.DataFrame(columns=['first_ratid', 'partner_ratid', 'group', 'group_geno', 'video_nake', 'is_blackfirst'],
                            data=rows_list)

    df_group['beh_key'] = pd.Series(['fps30_' + v.replace('-','_') + '_startFrame0_' + ('blackFirst' if is_blackfirst else 'whiteFirst')
                            for v, is_blackfirst in zip(df_group['video_nake'], df_group['is_blackfirst'])])

    # 对 Category 列进行分组并计数
    count = df_group.groupby('group').size()
    print(count)
    return df_group


def get_behavior_percentages(bhv, bhv_unique):
    return [np.mean(bhv==c) for c in bhv_unique] 


def define_group_freq(bhvSeqs, df_group, df_labnames):
    bhv_unique = np.sort(np.unique(df_labnames['behlabel']))
    freq_data = [get_behavior_percentages(np.array(bhvSeqs[beh_key]), bhv_unique)
                    for beh_key in df_group['beh_key']]
    freq_data = np.array(freq_data)
    # freq_data[:,1:] /= (1-freq_data[:,0])[:, None]
    df_freq_data_list = []
    for i, freq_data_this in enumerate(freq_data):
        df_freq_data = pd.DataFrame({'behlabel':range(len(freq_data_this)), 'freq':freq_data_this})
        df_freq_data['beh_key'] = df_group['beh_key'].iloc[i]
        df_freq_data_list.append(df_freq_data)

    df_freq_data_list = pd.concat(df_freq_data_list, axis=0)

    df_group_x_freq = pd.merge(pd.merge(df_group, df_freq_data_list, on='beh_key'), df_labnames, on='behlabel')
    return freq_data, df_group_x_freq


def plot_box_data(df_labnames, df_group_x_freq, hue_order, hue_legend, savefigpath=None, autosort=None):
    assert len(hue_order) == 2
    if autosort is None:
        df_group_x_freq_A = df_group_x_freq[df_group_x_freq['group']==hue_order[0]]
        df_group_x_freq_B = df_group_x_freq[df_group_x_freq['group']==hue_order[1]]
        behlabel_unique = df_group_x_freq_A['behlabel'].unique()
        lab_names_unique = df_group_x_freq_A['lab_names'].unique()
        fold_changes = [df_group_x_freq_A[df_group_x_freq_A['behlabel']==i]['freq'].mean() /
                        df_group_x_freq_B[df_group_x_freq_B['behlabel']==i]['freq'].mean() 
                            for i in behlabel_unique]
        fold_changes = np.array(fold_changes)
        fc_inds=np.argsort(fold_changes)[::-1]
        behlabel_sort = behlabel_unique[fc_inds]
        lab_names_sort = lab_names_unique[fc_inds]
    else:
        behlabel_sort =  autosort['behlabel'].values
        lab_names_sort = autosort['lab_names'].values

    plt.figure(figsize = (15,15))
    sns.boxplot(y='lab_names', x='freq', hue='group', 
                hue_order=hue_order,
                data= df_group_x_freq,
                width=0.45,
                order=lab_names_sort,
                orient='h')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Percentage', fontsize=20)
    plt.ylabel('Label', fontsize=20)
    leg = plt.legend(fontsize=14)
    for i, text in enumerate(hue_legend): leg.get_texts()[i].set_text(text)

    xcolors=[]
    tPs = []
    signs = []
    for behlabel in behlabel_sort:
        # behlabel = behlabel_sort[i]
        AB = df_group_x_freq[df_group_x_freq['behlabel']==behlabel]
        A = AB[AB['group']==hue_order[0]]
        B = AB[AB['group']==hue_order[1]]
        A_freq = np.sort(A['freq'].values)
        B_freq = np.sort(B['freq'].values)
        # A_freq = A_freq[1:-1]
        # B_freq = B_freq[1:-1]
        t,p = stats.mannwhitneyu(A['freq'], B['freq'])
        # t,p = stats.ttest_ind(A_freq, B_freq)
        signs.append(A_freq.mean() - B_freq.mean())
        tPs.append(p)

    sig_ifs,tPs2,_,_=multipletests(tPs,method='fdr_bh')
    for p, sign in zip(tPs2, signs):
        if p<0.05 and sign>0:
            xcolors.append('blue')
        elif p<0.05 and sign<0:
            xcolors.append('brown')
        else:
            xcolors.append('black')
    
    ax = plt.gca()
    for kbi in range(len(behlabel_sort)):
       ax.get_yticklabels()[kbi].set_color(xcolors[kbi])

    plt.ylim(plt.ylim()[::-1])
        
    if savefigpath:
        plt.savefig(savefigpath, bbox_inches='tight')



def plot_bar_data(df_labnames, df_group_x_freq, hue_order, hue_legend, savefigpath=None, autosort=None):
    assert len(hue_order) == 2
    if autosort is None:
        df_group_x_freq_A = df_group_x_freq[df_group_x_freq['group']==hue_order[0]]
        df_group_x_freq_B = df_group_x_freq[df_group_x_freq['group']==hue_order[1]]
        behlabel_unique = df_group_x_freq_A['behlabel'].unique()
        lab_names_unique = df_group_x_freq_A['lab_names'].unique()
        fold_changes = [df_group_x_freq_A[df_group_x_freq_A['behlabel']==i]['freq'].mean() /
                        df_group_x_freq_B[df_group_x_freq_B['behlabel']==i]['freq'].mean() 
                            for i in behlabel_unique]
        fold_changes = np.array(fold_changes)
        fc_inds=np.argsort(fold_changes)[::-1]
        behlabel_sort = behlabel_unique[fc_inds]
        lab_names_sort = lab_names_unique[fc_inds]
    else:
        behlabel_sort =  autosort['behlabel'].values
        lab_names_sort = autosort['lab_names'].values

    plt.figure(figsize = (15,15))
    sns.barplot(y='lab_names', x='freq', hue='group', 
                hue_order=hue_order,
                data= df_group_x_freq,
                errorbar='se',
                width=0.6,
                order=lab_names_sort,
                orient='h')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Percentage', fontsize=20)
    plt.ylabel('Label', fontsize=20)
    leg = plt.legend(fontsize=14)
    for i, text in enumerate(hue_legend): leg.get_texts()[i].set_text(text)
    xcolors=[]
    tPs = []
    signs = []
    for behlabel in behlabel_sort:
        # behlabel = behlabel_sort[i]
        AB = df_group_x_freq[df_group_x_freq['behlabel']==behlabel]
        A = AB[AB['group']==hue_order[0]]
        B = AB[AB['group']==hue_order[1]]
        A_freq = np.sort(A['freq'].values)
        B_freq = np.sort(B['freq'].values)
        # A_freq = A_freq[1:-1]
        # B_freq = B_freq[1:-1]
        t,p = stats.mannwhitneyu(A['freq'], B['freq'])
        # t,p = stats.ttest_ind(A_freq, B_freq)
        signs.append(A_freq.mean() - B_freq.mean())
        tPs.append(p)

    sig_ifs,tPs2,_,_=multipletests(tPs,method='fdr_bh')
    for p, sign in zip(tPs2, signs):
        if p<0.05 and sign>0:
            xcolors.append('blue')
        elif p<0.05 and sign<0:
            xcolors.append('brown')
        else:
            xcolors.append('black')
    
    ax = plt.gca()
    for kbi in range(len(behlabel_sort)):
       ax.get_yticklabels()[kbi].set_color(xcolors[kbi])

    plt.ylim(plt.ylim()[::-1])
        
    if savefigpath:
        plt.savefig(savefigpath, bbox_inches='tight')


def plot_diff_box_data(df_group_x_freq, hue_order):
    assert len(hue_order) == 2
    df_group_x_freq_A = df_group_x_freq[df_group_x_freq['group']==hue_order[0]]
    df_group_x_freq_B = df_group_x_freq[df_group_x_freq['group']==hue_order[1]]
    behlabel_unique = df_group_x_freq_A['behlabel'].unique()
    lab_names_unique = df_group_x_freq_A['lab_names'].unique()
    A = np.array([df_group_x_freq_A[df_group_x_freq_A['behlabel']==i]['freq'].mean() for i in behlabel_unique])
    B = np.array([df_group_x_freq_B[df_group_x_freq_B['behlabel']==i]['freq'].mean() for i in behlabel_unique])
    fold_changes = (B-A)/(A+B+0.001)
    fc_inds=np.argsort(fold_changes)[::-1]
    behlabel_sort = behlabel_unique[fc_inds]
    lab_names_sort = lab_names_unique[fc_inds]
    lab_names_sort = [i.split(' [')[0] for i in lab_names_sort]
    freq_ratio_sort = fold_changes[fc_inds]
    
    xcolors=[]
    for behlabel in behlabel_sort:
        AB = df_group_x_freq[df_group_x_freq['behlabel']==behlabel]
        A = AB[AB['group']==hue_order[0]]
        B = AB[AB['group']==hue_order[1]]
        A_freq = np.sort(A['freq'].values)
        B_freq = np.sort(B['freq'].values)
        t,p = stats.ttest_ind(A_freq, B_freq)
        if p<0.05 and A_freq.mean()>B_freq.mean():
                xcolors.append('blue')
        elif p<0.05 and A_freq.mean()<B_freq.mean():
                xcolors.append('brown')
        else:
                xcolors.append('black')
    
    plt.figure(figsize=(6,9))
    plt.barh(np.arange(len(freq_ratio_sort)), 
            freq_ratio_sort,
            facecolor = [0.5, 0.5, 0.5])
    plt.xlim([-0.6, 0.6])
    plt.ylim([-1, len(freq_ratio_sort)])
    plt.plot([0, 0], [-0.5, len(freq_ratio_sort)-0.5], 'k')
    plt.xticks([-0.6, 0, 0.6])
    plt.yticks(np.arange(len(freq_ratio_sort)), lab_names_sort)
    plt.xlabel('Differential expression', fontsize=16)
    
    ind_in = np.array(xcolors)=='brown'
    plt.barh(np.arange(len(freq_ratio_sort))[ind_in], 
            freq_ratio_sort[ind_in],
            facecolor = '#dd1c77')
    
    ind_in = np.array(xcolors)=='blue'
    plt.barh(np.arange(len(freq_ratio_sort))[ind_in], 
            freq_ratio_sort[ind_in],
            facecolor = '#3182bd')
    
    plt.figure(figsize=(9,4))
    plt.bar(np.arange(len(freq_ratio_sort)), 
            freq_ratio_sort,
            facecolor = [0.5, 0.5, 0.5])
    plt.ylim([-0.8, 0.8])
    plt.xlim([-1, len(freq_ratio_sort)])
    plt.plot([-0.5, len(freq_ratio_sort)-0.5], [0, 0], 'k')
    plt.yticks([-1, 0, 1])
    plt.xticks(np.arange(len(freq_ratio_sort)), lab_names_sort, 
            rotation=30, ha='right')
    plt.ylabel('Differential expression', fontsize=16)
    
    ind_in = np.array(xcolors)=='brown'
    plt.bar(np.arange(len(freq_ratio_sort))[ind_in], 
            freq_ratio_sort[ind_in],
            facecolor = '#dd1c77')
    
    ind_in = np.array(xcolors)=='blue'
    plt.bar(np.arange(len(freq_ratio_sort))[ind_in], 
            freq_ratio_sort[ind_in],
            facecolor = '#3182bd')
    plt.xlim(plt.xlim()[::-1])

def plt_ellipse(x, y, ax, color):
    data = np.array([x, y])
    pca_ = PCA(n_components=2)
    pca_.fit(data.T)
    components = pca_.components_
    rotation_angle = np.arctan2(components[1, 0], components[0, 0])
    rotation_angle_degrees = np.degrees(rotation_angle)
    std_deviations = np.sqrt(pca_.explained_variance_)

    theta = np.linspace(0, 2*np.pi, 100)
    x = std_deviations[0] * np.cos(theta)
    y = std_deviations[1] * np.sin(theta)
    ellipse = np.vstack((x, y)).T.dot(components)*2 + np.mean(data, axis=1)
    ax.plot(ellipse[:, 0], ellipse[:, 1], color=color)


def plt_ellipsoid(x, y, z, ax, color):
    data = np.array([x, y, z])
    pca_ = PCA(n_components=3)
    pca_.fit(data.T)
    A = pca_.get_covariance()

    center = np.array([np.mean(x), np.mean(y), np.mean(z)])
    U, s, rotation = linalg.svd(A)
    radii = np.sqrt(s) * 2
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 60)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

    ax.plot_surface(x, y, z,  rstride=3, cstride=3,  color=color, linewidth=0.1, alpha=0.2, shade=True)



def plot_dim_reduced_2d(data_dim_reduced, data_df, xylabel, list_geno, legend_list_geno, colors, ax=None, savefigpath=None):
    data_df['x'] = data_dim_reduced[:,0]
    data_df['y'] = data_dim_reduced[:,1]
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
    
    for c, geno, geno_label in zip(colors, list_geno, legend_list_geno):
        print(c, geno)
        data_geno = data_df[data_df['group']==geno]
        x, y = data_geno['x'], data_geno['y']
        plt_ellipse(x, y, ax, c)
        ax.scatter(x, y, c=c, s=60, label=geno_label)

    plt.legend(legend_list_geno, fontsize=16)
    ax.set_xticks([])
    # plt.yticks([-10])
    ax.set_yticks([])
    ax.set_xlabel(f'{xylabel}-1', fontsize=20)
    ax.set_ylabel(f'{xylabel}-2', fontsize=20)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if savefigpath:
        plt.savefig(savefigpath, bbox_inches='tight')



def plot_dim_reduced_3d(data_dim_reduced, data_df, xylabel, list_geno, legend_list_geno, colors, ax=None, savefigpath=None):
    data_df['LDA_1'] = data_dim_reduced[:, 0]
    data_df['LDA_2'] = data_dim_reduced[:, 1]
    data_df['LDA_3'] = data_dim_reduced[:, 2]

    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
    
    for c, geno, genolabel in zip(colors, list_geno, legend_list_geno):
        print(c, geno)
        data_geno = data_df[data_df['group']==geno]
        x, y, z = data_geno['LDA_1'].values, data_geno['LDA_2'].values, data_geno['LDA_3'].values
        plt_ellipsoid(x, y, z, ax, c)
        ax.scatter(x, y, z, c=c, s=60, label=genolabel)

    # plt.xticks([])
    # plt.yticks([-10])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel(f'{xylabel}-1', fontsize=20)
    ax.set_ylabel(f'{xylabel}-2', fontsize=20)
    ax.set_zlabel(f'{xylabel}-3', fontsize=20)
    plt.legend(loc="upper right", fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.view_init(azim=-65)   # 俯视(azim=270, elev=90) 

    if savefigpath:
        plt.savefig(savefigpath, bbox_inches='tight')


def create_nodemerge_group_x_freq(df_group_x_freq, cluster_nodes_merged):
    node_from, node_to, node_toname = zip(*cluster_nodes_merged)
    node_from_np = np.array(node_from)
    node_to_np = np.array(node_to)
    node_toname_np = np.array(node_toname)
    list_map = {}
    for i in np.unique(node_to_np):
        node_from_list = np.sort(node_from_np[node_to_np==i])
        node_toname_postfix  = node_toname_np[node_to_np==i][0] + ' [{}]'.format(','.join([str(i) for i in node_from_list]))
        list_map[i] = (node_from_list, node_toname_postfix)

    df_group_x_freq_groupmerge = df_group_x_freq.copy()
    for i in np.unique(node_to_np):
        node_from_list, node_toname_postfix = list_map[i]
        subset = df_group_x_freq['behlabel'].isin(node_from_list)
        df_group_x_freq_groupmerge['behlabel'][subset] = i
        df_group_x_freq_groupmerge['lab_names'][subset]= node_toname_postfix

    return df_group_x_freq_groupmerge

def get_clean_df(df_group_x_freq, noutlier=1):
    df_clean_list = []
    for _, df in df_group_x_freq.groupby('group'):
        df_clean_list.append(df.groupby('behlabel').apply(
                            lambda x: x.nsmallest(len(x)-noutlier, 'freq').nlargest(len(x)-2*noutlier, 'freq')))
    df_clean = pd.concat(df_clean_list)
    return df_clean

# %%
if __name__ == '__main__':
    rat_info, video_info, bhvSeqs, df_labnames, k_best, cluster_nodes_merged = load_data(project, sheet_name)
    df_group = define_group(rat_info, video_info, df_labnames)
    freq_data, df_group_x_freq = define_group_freq(bhvSeqs, df_group, df_labnames)
    df_group_x_freq_groupmerge = create_nodemerge_group_x_freq(df_group_x_freq, cluster_nodes_merged)

    df_group_x_freq_cp = df_group_x_freq.copy()
    df_group_x_freq_groupmerge = df_group_x_freq_groupmerge.copy()
    df_group_x_freq = get_clean_df(df_group_x_freq)
    df_group_x_freq_groupmerge = get_clean_df(df_group_x_freq_groupmerge)

    hue_order = ['malemale', 'femalefemale']
    hue_legend = ['M_M', 'F_F']
    plot_box_data(df_labnames, df_group_x_freq, hue_order, hue_legend, 
                savefigpath=osp.join(project, 'boxplot_MM_vs_FF.pdf'))
    plot_box_data(df_labnames, df_group_x_freq_groupmerge, hue_order, hue_legend,
                savefigpath=osp.join(project, 'boxplot_MM_vs_FF_nodemerge.pdf'))

    hue_order = ['malefemale', 'femalemale']
    hue_legend = ['[M]_F', '[F]_M']
    plot_box_data(df_labnames, df_group_x_freq, hue_order, hue_legend,
                savefigpath=osp.join(project, 'boxplot_MF_vs_FM.pdf'))
    plot_box_data(df_labnames, df_group_x_freq_groupmerge, hue_order, hue_legend,
                savefigpath=osp.join(project, 'boxplot_MF_vs_FM_nodemerge.pdf'))

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

# %%
