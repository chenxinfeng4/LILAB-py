# python -m lilab.openlabcluster_postprocess.s3b_hiecluster_plot A/B/C.clippredpkl
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


plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

#%%

def main(clippredpkl_file):
    clippreddata = pickle.load(open(clippredpkl_file, "rb"))
    assert {
        "ncluster",
        "ntwin",
        "cluster_labels",
        "embedding",
        "embedding_d2",
        "clipNames",
        "cluster_names",
    } <= clippreddata.keys()
    # %% 载入数据
    feat_data = clippreddata["embedding"]  # nsample x nfeat

    # start from 1. The 0 is nonsocial
    kmeans_label = clippreddata["cluster_labels"]
    n_kmeans = clippreddata["ncluster"]
    cluster_names = clippreddata['cluster_names']
    assert kmeans_label.min() == 1

    if False:
        # 归一化处理
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(feat_data)

        # PCA降维
        pca = PCA(n_components=30)  # 选择主成分的个数
        condensed_data = pca.fit_transform(data_scaled)
    else:
        condensed_data = feat_data

    # Kmeans 聚类的中心点
    cluster_centers = np.zeros(
        (n_kmeans, condensed_data.shape[1])
    )  # nkmeans x nfeat，每一行是一个类的中心
    for i in range(n_kmeans):
        cluster_centers[i] = condensed_data[kmeans_label == i+1].mean(axis=0)

    # %% 谱系聚类
    # 可选 cosine 和 euclidean, complete 和 average
    # dist = pdist(cluster_centers)
    dist = pdist(cluster_centers, metric="cosine")
    # linkage_matrix = linkage(dist, method='average')
    linkage_matrix = linkage(dist, method="complete")
    fig, ax = plt.subplots(figsize=(4, 14))
    plt.rcParams["font.family"] = "Arial"
    leaf_labels = [
        f"[{leaf+1:>2}] {label.rstrip()}" for leaf, label in enumerate(cluster_names)
    ]
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
    plt.xticks([])
    plt.yticks(fontsize=14)

    outfig = osp.join(osp.dirname(clippredpkl_file), "dendrogram.pdf")
    plt.savefig(outfig, bbox_inches="tight")

    leaf_order = dendrogram_result['leaves']
    clippreddata['cluster_labels_ordered'] =  np.array(leaf_order) + 1  #nonsocial 不在里面
    pickle.dump(clippreddata, open(clippredpkl_file, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clippredpkl", type=str)
    args = parser.parse_args()
    assert osp.isfile(args.clippredpkl)
    main(args.clippredpkl)
