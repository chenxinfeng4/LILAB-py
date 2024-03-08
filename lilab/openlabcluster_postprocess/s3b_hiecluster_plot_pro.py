# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os.path as osp
import argparse
import scipy.stats
import pandas as pd
import seaborn as sns
import sys
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

#%%
# clippredpkl_file = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/00_BehaviorAnalysis-seq2seq/miniscope-behavior/ID010/20230809/FWPCA0.00_P100_en3_hid30_epoch250-decSeqPC0.9_svm2allAcc0.91_kmeansK2use-45_fromK1-20_K100.clippredpkl'
clippredpkl_file = sys.argv[1]
beh_project_dir = osp.dirname(clippredpkl_file)
clippreddata = pickle.load(open(clippredpkl_file, "rb"))
assert {
    "ncluster",
    "ntwin",
    "cluster_labels",
    "embedding",
    "embedding_d2",
    "clipNames"
} <= clippreddata.keys()
# %% 载入数据
feat_data = clippreddata["embedding"]  # nsample x nfeat

# start from 1. The 0 is nonsocial
kmeans_label = clippreddata["cluster_labels"]
n_kmeans = clippreddata["ncluster"]
cluster_names = clippreddata['cluster_names']
assert kmeans_label.min() == 1

def get_pearsonr(xs, ys, embedding):
    return np.array([scipy.stats.pearsonr(embedding[x], embedding[y])[0] for x, y in zip(xs, ys)])



n_each_class = 200
n_blockin = 4
clusterlabel = clippreddata["cluster_labels"] # kmeans_data['labels'] + 1
assert clusterlabel.min()==1 and clusterlabel.max()==n_kmeans
uni_id, uni_c = np.unique(clusterlabel, return_counts=True)
np.random.seed(1000)

n_each_class = np.min([uni_c.min()//2, 200])

id_ego_choose_each_class = np.zeros((len(uni_id)+1, n_each_class, n_blockin), dtype=int)
id_partner_choose_each_class = np.zeros_like(id_ego_choose_each_class)
id_nchoose_each_class = np.zeros_like(id_ego_choose_each_class)
for i in uni_id:
    id_list = np.where(clusterlabel==i)[0]

    id_choose = np.stack([np.random.choice(id_list, n_blockin*2, replace=True) for _ in range(n_each_class)])
    id_ego_choose_each_class[i] = id_choose[...,  :n_blockin]
    id_partner_choose_each_class[i] = id_choose[...,  n_blockin:]

    id_list = np.where(clusterlabel!=i)[0]
    id_nchoose = np.stack([np.random.choice(id_list, n_blockin, replace=True) for _ in range(n_each_class)])
    id_nchoose_each_class[i] = id_nchoose


df = pd.DataFrame(columns=['r', 'group', 'cluster'])
df_list = []
for i in uni_id:
    xego = id_ego_choose_each_class[i]
    xpartner = id_partner_choose_each_class[i]
    yother = id_nchoose_each_class[i]
    rs_intra = [get_pearsonr(xego_, xpartner_, feat_data).mean() for (xego_, xpartner_) in zip(xego, xpartner)]
    rs_inter = [get_pearsonr(xego_, yother_, feat_data).mean() for (xego_, yother_) in zip(xego, yother)]
    df1 = pd.DataFrame(zip(rs_intra, ['intra']*n_each_class, np.zeros((n_each_class,), dtype=int)+i))
    df2 = pd.DataFrame(zip(rs_inter, ['inter']*n_each_class, np.zeros((n_each_class,), dtype=int)+i))
    df_list.extend([df1, df2])


df = pd.concat(df_list)
df.columns = ['r', 'group', 'cluster']
df_mean = df.groupby(by=['group', 'cluster']).agg('median').reset_index()
df_mean.loc[(df_mean['group'] =='intra'), 'cluster'] -= 1.2
df_mean.loc[(df_mean['group'] =='inter'), 'cluster'] -= 0.8


plt.figure(figsize= (20,8))
palette = ['#a4cee4', '#e6e6e6']
sns.violinplot(data=df, x='cluster', y='r', scale='count', 
               palette=palette, cut = 5,
               inner=None, hue='group')

plt.plot(df_mean['cluster'].values, df_mean['r'].values, 'wo', markeredgecolor='k')
plt.xlim(-0.5, n_kmeans - 0.5)
plt.ylim(-1, 1)
plt.yticks([-1, -0.5, 0, 0.5, 1])
plt.ylabel('Correlation coefficient')
outfig = osp.join(beh_project_dir, 'violinplot_cluster_similarity_plot.pdf')
plt.savefig(outfig)
#%%
K = len(uni_id)
matrix = np.zeros((K,K))
for i in range(K):
    for j in range(K):
        xego = id_ego_choose_each_class[i+1][...,0]
        xpartner = id_partner_choose_each_class[j+1][...,0]
        matrix[i, j] = get_pearsonr(xego, xpartner, feat_data).mean()

matrix_mirror = (matrix + matrix.T)/2

leaf_labels = [
    f"[{leaf+1:>2}] {label.rstrip()}" for leaf, label in enumerate(cluster_names)
]

df_matrix_mirror = pd.DataFrame(matrix_mirror, columns=np.arange(1,K+1), index=np.arange(1,K+1))
df_matrix_mirror = pd.DataFrame(matrix_mirror, columns=np.arange(1,K+1), index=leaf_labels)
sns.clustermap(df_matrix_mirror, cmap="vlag", vmin=-1, vmax=1,
               xticklabels=1,yticklabels=1,
               method='complete',
               figsize=(11,8) )
outfig = osp.join(osp.dirname(clippredpkl_file), "dendrogram_heatmap_similiar.pdf")
plt.savefig(outfig, bbox_inches="tight")

dist = pdist(matrix_mirror)
linkage_matrix = linkage(dist, 
                         method='complete',
                         )


fig, ax = plt.subplots(figsize=(4, 14))
plt.rcParams["font.family"] = "Arial"
dendrogram_result = dendrogram(
    linkage_matrix,
    ax=ax,
    labels=leaf_labels,
    orientation="left",
    color_threshold=0,
    above_threshold_color="k",
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_ylim(ax.get_ylim()[::-1])
plt.xticks([])
plt.yticks(fontsize=14)

outfig = osp.join(osp.dirname(clippredpkl_file), "dendrogram_similiar.pdf")
plt.savefig(outfig, bbox_inches="tight")