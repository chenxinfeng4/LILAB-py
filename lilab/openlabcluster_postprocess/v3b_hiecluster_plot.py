# python -m lilab.openlabcluster_postprocess.v3b_hiecluster_plot xx
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
from lilab.openlabcluster_postprocess.s1_merge_3_file import get_assert_1_file
import radialtree as rt
from lilab.openlabcluster_postprocess.v2b_usv_tsne_masic_shape import get_the_centered, get_cmap
from PIL import Image


plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

clippredpkl_file = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/shank3-USV_all/reconstruction/svm2allAcc0.9_kmeansK2use-49_fromK1-20_K100.usvclippredpkl'


def main(clippredpkl_file):
    project_dir = osp.dirname(clippredpkl_file)
    clippreddata = pickle.load(open(clippredpkl_file, "rb"))
    assert {
        "ncluster",
        "cluster_labels",
        "embedding",
        "embedding_d2",
        "clipNames",
    } <= clippreddata.keys()
    feat_data = clippreddata["embedding"]  # nsample x nfeat

    kmeans_label = clippreddata["cluster_labels"]
    n_kmeans = clippreddata["ncluster"]
    cluster_names = ['' for _ in range(n_kmeans)]
    assert kmeans_label.min() == 0

    # 归一化处理
    scaler = StandardScaler()
    condensed_data = scaler.fit_transform(feat_data)

    # PCA降维
    if False:
        pca = PCA(n_components=6)  # 选择主成分的个数
        condensed_data = pca.fit_transform(condensed_data)

    # Kmeans 聚类的中心点
    cluster_centers = np.zeros(
        (n_kmeans, condensed_data.shape[1])
    )  # nkmeans x nfeat，每一行是一个类的中心
    for i in range(n_kmeans):
        cluster_centers[i] = condensed_data[kmeans_label == i].mean(axis=0)

    dist = pdist(cluster_centers, metric="cosine")
    # linkage_matrix = linkage(dist, method='average')
    linkage_matrix = linkage(dist, method="complete")
    fig, ax = plt.subplots(figsize=(4, 14))
    plt.rcParams["font.family"] = "Arial"
    leaf_labels = [
        f"[{leaf:>2}] {label.rstrip()}" for leaf, label in enumerate(cluster_names)
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

    outfig = osp.join(osp.dirname(clippredpkl_file), "usv_dendrogram.pdf")
    plt.savefig(outfig, bbox_inches="tight")

    leaf_order = dendrogram_result['leaves']
    clippreddata['cluster_labels_ordered'] =  np.array(leaf_order)
    pickle.dump(clippreddata, open(clippredpkl_file, "wb"))
    dendrogram_result['color_list'] = ['C0']*n_kmeans
    
    rt.plot(dendrogram_result)
    plt.savefig(osp.join(osp.dirname(clippredpkl_file), "usv_dendrogram_circle.pdf"))

    if False:
        # get the most center samples
        center_ids = np.array([get_the_centered(feat_data, np.where(kmeans_label==i)[0])
                                for i in range(n_kmeans)])
        center_clipNames = clippreddata['clipNames'][center_ids]
        center_files = [osp.join(project_dir, video_nake, f'{idx_in_file+1:06d}.png') #start from 1
                            for video_nake, idx_in_file in center_clipNames]

        from IPython.display import display

        pil_mask, pil_mask_rgb = get_cmap() #size 128x128
        pil_legend_rgb = pil_mask_rgb[:,0,:]
        dendrogram_colors = np.zeros((n_kmeans, 3), dtype=float)

        imgs_all = np.empty((n_kmeans,), dtype=object)
        for i, imgfile in enumerate(center_files):
            img = Image.open(imgfile).crop((100, 50, 160, 200)).resize(pil_mask.size)
            img_np = np.array(img.convert('RGB'))[...,0] #to gray
            img_np[64,64] = 255
            img_bool = img_np>10
            main_freq_ind = np.sum(np.sum(img_bool,axis=1)/np.sum(img_bool) * np.arange(img_np.shape[0]))
            color = pil_legend_rgb[int(main_freq_ind)]
            dendrogram_colors[i, :3] = color
            imgs_all[i] = img
            # plt.imshow(img_np)
            # plt.plot([0, 128], [main_freq_ind, main_freq_ind], color=color/255, linewidth=2)
            # plt.show()
            # display(img)
        dendrogram_colors /= 255
        colors_dict={"Main frequency":dendrogram_colors}

        rt.plot(dendrogram_result, colorlabels=colors_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('usvclippredpkl', type=str)
    args = parser.parse_args()
    assert osp.isfile(args.usvclippredpkl)
    main(args.usvclippredpkl)
