# python -m lilab.openlabcluster_postprocess.s2_decisionBoundary_masaik_plot xxx.clippredpkl
"""
conda activate mmdet
"""
# %%
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from sklearn.manifold import TSNE
import umap
import os.path as osp
import pickle
from lilab.feature_autoenc.heatmap_functions import HeatmapType, heatmap
from lilab.comm_signal.line_scale import line_scale
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import LinearSegmentedColormap
import argparse

# matplotlib.use("Agg")

clippredpkl_file = "/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/DayAll20230828/All-DecSeq/FWPCA0.00_P100_en3_hid30_epoch300-decSeqPC0.9_svm2allAcc0.96_kmeansK2use-43_fromK1-20_K100.clippredpkl"

umap_mask_threshold = 0.5
nbin = 150

#%%
def center_of_mass(map2D: np.ndarray):
    map2D_bool = np.zeros_like(map2D, dtype=bool)
    map2D_bool[map2D > 0] = True
    if np.sum(map2D_bool) == 0:
        return np.nan, np.nan

    a_x, a_y = (
        np.sum(map2D_bool, axis=0, keepdims=True),
        np.sum(map2D_bool, axis=1, keepdims=True),
    )
    a_all = np.sum(a_x)
    a_x, a_y = a_x / a_all, a_y / a_all
    grids = np.ogrid[[slice(0, i) for i in map2D.shape]]
    return np.sum(a_y * grids[0]), np.sum(a_x * grids[1])


def center_of_mass_cv(map2D: np.ndarray):
    map2D_bool = np.zeros_like(map2D, dtype=bool)
    map2D_bool[map2D > 0] = True
    if np.sum(map2D_bool) <= map2D.size * 0.005:
        return np.nan, np.nan

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        map2D_bool.astype(np.uint8)
    )
    max_area_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    center_x = int(centroids[max_area_idx][0])
    center_y = int(centroids[max_area_idx][1])
    return center_y, center_x


def plot_knn_decision_boundary_mosaik(heatobj, X, y, clu_lab_uni, clu_lab_shuf, n_neighbors=100):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, nbin), np.linspace(x_min, x_max, nbin)
    )
    xx_i, yy_i = np.meshgrid(np.arange(nbin), np.arange(nbin))

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y)
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    xxyy_i = np.c_[xx_i.ravel(), yy_i.ravel()]

    z0 = knn.predict(xxyy)  # shape:[meshgrid两个shape相乘，2]
    z = z0.reshape(xx.shape)  # [x*y] -> [x, y]. 这里z就是对应点的分类效果
    zc = z  ##from 2 to differ from scatter
    # add mosaik
    classfimap = np.zeros((nbin, nbin))
    for i, (x_i, y_i) in enumerate(xxyy_i):
        classfimap[x_i, y_i] = z0[i]
    classfimap_shuf = np.zeros_like(classfimap)
    for idsrc, idtrg in zip(clu_lab_uni, clu_lab_shuf):
        classfimap_shuf[classfimap == idsrc] = idtrg

    heatobj.classfimap = classfimap.astype(int)
    heatobj.classfimap_shuf = classfimap_shuf

    # heatobj.plt_heatmap("classfimap_shuf")
    # heatobj.plt_heatmap()
    heatobj.plt_heatmap("classfimap")
    heatobj.plt_maskheatmap()
    # plt.axis('off')
    ##add contour
    cNOs = list(set(y))
    masked_heatmap = heatobj.density_mask * heatobj.classfimap
    centers = np.array([center_of_mass_cv(masked_heatmap == i) for i in cNOs])
    cNOs_dicts = {}
    for i in cNOs:  # from 1
        cNOs_dicts[i] = centers[i - 1]

    print(cNOs_dicts)

    for i, c in enumerate(cNOs):
        plt.text(
            cNOs_dicts[c][0],
            cNOs_dicts[c][1],
            str(c),
            fontsize=15,
            horizontalalignment="center",
            verticalalignment="center",
        )
    # plt.gca().set_aspect('equal', 'datalim')
    ##

    cMax = np.max(y) + 1
    # plt.colorbar(boundaries=np.arange(1,cMax)).set_ticks(np.arange(1,cMax))
    plt.xticks([])
    plt.yticks(
        [-100,]
    )
    # plt.yticks([0, 200])
    plt.xlabel("tSNE-1", fontsize=24)
    plt.ylabel("tSNE-2", fontsize=24)

    ax = plt.gca()
    # set spine line to invisible
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.tight_layout()


def main(clippredpkl_file, outfig=None):
    clippreddata = pickle.load(open(clippredpkl_file, "rb"))
    assert {
        "ncluster",
        "cluster_labels",
        "embedding",
        "embedding_d2",
    } <= clippreddata.keys()

    embedding = clippreddata["embedding_d2"]
    clu_labs = clippreddata["cluster_labels"]
    #%%
    if True:
        embedded_d2_umap = clippreddata["embedding_d2"]
    else:
        tsne = TSNE(n_components=2, random_state=42)
        embedded_d2_tsne = tsne.fit_transform(clippreddata["embedding"])

        umap_model = umap.UMAP(n_components=2, random_state=42)
        embedded_d2_umap = umap_model.fit_transform(clippreddata["embedding"])

    cmap = LinearSegmentedColormap.from_list(
        name="white_cmap", colors=[(1, 1, 1), (1, 1, 1)]
    )

    clu_lab_uni = np.unique(clu_labs)
    clu_lab_shuf = clu_lab_uni.copy()
    np.random.seed(2)
    np.random.shuffle(clu_lab_shuf)
    # plot_knn_decision_boundary_mosaik(embedding, clu_labs, mosaikFig,k=kNum)

    # heatmat, xedges, yedges = heatmap(embedding, bins=nbin)
    if False:
        embedded_d2 = embedded_d2_tsne
    else:
        embedded_d2 = embedded_d2_umap
    heatmat, xedges, yedges = heatmap(embedded_d2, bins=nbin)

    plt.figure(figsize=(20, 10))
    heatobj = HeatmapType(heatmat, xedges, yedges)
    heatobj.calculate()
    heatobj.plt_heatmap()
    # heatobj.plt_boundary()
    heatobj.set_mask_by_densitymap(umap_mask_threshold)

    # %%
    # 把 embeding 的 xy坐标系映射到 bin 的像素坐标系
    embedding_re_XY = np.zeros_like(embedding)
    embedding_re_XY[:, 0] = line_scale(
        [embedding[:, 0].min(), embedding[:, 0].max()], [0, len(xedges)], embedding[:, 0]
    )
    embedding_re_XY[:, 1] = line_scale(
        [embedding[:, 1].min(), embedding[:, 1].max()], [0, len(yedges)], embedding[:, 1]
    )

    X, y = embedding_re_XY, clu_labs
    print('Begin to plot figure')
    plt.figure(figsize=(10, 10))
    plot_knn_decision_boundary_mosaik(heatobj, X, y, clu_lab_uni, clu_lab_shuf, n_neighbors=100)
    if not outfig: outfig = osp.join(osp.dirname(clippredpkl_file), 'kmeans_umap.pdf')
    print('save to ', outfig)
    plt.savefig(outfig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('clippredpkl', type=str)
    args = parser.parse_args()
    assert osp.isfile(args.clippredpkl)
    main(args.clippredpkl)
