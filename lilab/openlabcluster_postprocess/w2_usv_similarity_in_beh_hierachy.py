#%%
import pickle
import numpy as np
import scipy
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as osp

pklfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/DayAll20230828/All-DecSeq_far_close/usv_beh_time_aligned_onset.pkl'
project_dir = osp.dirname(pklfile)
# %%
pkldata = pickle.load(open(pklfile, 'rb'))


def get_hist(data_cell):
    n = len(data_cell)
    data_flat = np.concatenate(data_cell)
    c, e = np.histogram(data_flat, bins=50)
    smooth_data = savgol_filter(c, 13, 2)
    smooth_data /= n
    return smooth_data


def get_hist_ego_other(data_cell):
    np.random.seed(100)
    ind_choose = np.arange(len(data_cell))
    np.random.shuffle(ind_choose)
    ind_ego = ind_choose[:len(ind_choose)//2]
    ind_other = ind_choose[len(ind_choose)//2:]
    data_ego = [data_cell[i] for i in ind_ego]
    data_other = [data_cell[i] for i in ind_other]
    return np.array([get_hist(data_ego), get_hist(data_other)])

usv_evt_dict = pkldata['usv_evt_dict']
data_cell = usv_evt_dict[(0, 0)]

behlabel_unique = pkldata['behlabel_unique']
usvlabel_unique = pkldata['usvlabel_unique']


smooth_data_ = get_hist(usv_evt_dict[(0, 0)])
nTick = len(smooth_data_)
kUSV = len(usvlabel_unique)
kBeh = len(behlabel_unique)
smooth_dataall = np.zeros((2, len(behlabel_unique), kUSV, nTick), dtype=float)

for beh_i in behlabel_unique:
    for usvlabel_i in usvlabel_unique:
        usv_evt_now = usv_evt_dict[(usvlabel_i, beh_i)]
        smooth_dataall[:, beh_i, usvlabel_i] = get_hist_ego_other(usv_evt_now)

smooth_dataall_mean = np.array([smooth_dataall[:,:,i].mean() for i in range(smooth_dataall.shape[2])])
smooth_dataall_norm = smooth_dataall / smooth_dataall_mean[None,None,:,None] - 1
# %%
tticks = np.linspace(-10, 10, nTick)
# twin = [-1, 1]
twin = [0, 0.8]
ind_in_twin = (tticks>twin[0]) & (tticks<twin[1])
smooth_datawithin = smooth_dataall_norm[..., ind_in_twin]


matrix = np.zeros((kBeh, kBeh))
for iBeh in range(kBeh):
    for jBeh in range(kBeh):
        ego   = smooth_datawithin[0, iBeh].ravel()
        other = smooth_datawithin[1, jBeh].ravel()
        matrix[iBeh, jBeh] = scipy.stats.pearsonr(ego, other)[0]
#%%
matrix_mirror = (matrix + matrix.T) / 2
#%%
beh_order = np.array(pkldata['beh_order'])
beh_order_non0 = beh_order[beh_order!=0]
matrix_sort = matrix_mirror[beh_order_non0][:,beh_order_non0]

behlabel_names = np.array([s[-4:] + ' ' + s[:-5] for s in pkldata['behlabel_names']])[beh_order_non0]
plt.figure(figsize=(14,8))
plt.imshow(matrix_sort, cmap="coolwarm")
plt.xticks(np.arange(len(beh_order_non0)), beh_order_non0, fontsize=6)
plt.yticks(np.arange(len(beh_order_non0)), behlabel_names)
plt.gca().yaxis.tick_right()
plt.title('USV similarity along behavior dendrogram')
plt.clim([-1, 1])
plt.tight_layout()
# plt.savefig(osp.join(project_dir, 'USV_similarity_along_behavior_dendrogram_offset.pdf'))
# %%
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import pandas as pd

dist = pdist(matrix_mirror)
linkage_matrix = linkage(dist, 
                         method='complete',
                         )
leaf_labels = np.array([s[-4:] + ' ' + s[:-5] for s in pkldata['behlabel_names']])
# df_matrix_mirror = pd.DataFrame(matrix_mirror, columns=np.arange(1,K+1), index=np.arange(1,K+1))
df_matrix_mirror = pd.DataFrame(matrix_mirror, columns=np.arange(len(leaf_labels)), index=leaf_labels)
sns.clustermap(df_matrix_mirror, cmap="vlag", vmin=-1, vmax=1,
               xticklabels=1,yticklabels=1,
               method='complete',
               figsize=(11,8) )
plt.savefig(osp.join(project_dir, f'USV_similarity_along_behavior_dendrogram_onset_t{twin[0]}_t{twin[1]}.pdf'))
#%%
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

#%%
outfig = osp.join(project_dir, f"dendrogram_similiar_usv_onset_t{twin[0]}_t{twin[1]}.pdf")

#%%
plt.savefig(outfig, bbox_inches="tight")
