# 拿到label之后，做一些图
陈昕枫，文档，2023-7-22

在拿 **OpenLabCluster** 输出的Cluster label数据之后，可以按照流程绘制下面的图。

## 1. 合并文件
首先将多个分散的数据合并。合并 `*_simple.hlpkl`, `clipNames.txt` 和 `*svm2all*.npz`。
- *_simple.hlpkl： {'encFeats', 'embedding', 'semilabel'}
- *svm2all*.npz： {'labels'} 片段的结果
- clipNames.txt:  list of clipnames

合并后 `*.clippredpkl`:
> cd PROJECT_DIR
>
> python -m lilab.openlabcluster_postprocess.s1_merge_3_file $PWD
```python
{
    'ncluster': ncluster,
    'ntwin': segLength,
    'cluster_labels': pred_labels,
    'embedding': embedding_d60,
    'embedding_d2': embedding_d2,
    'clipNames': np.array(clipNames),
}
```

## 2. Embedding 的 UMAP 与 Kmeans 边界
![UMAP 边界](images/openlabcluster_boundary_umap.jpg)

> python -m lilab.openlabcluster_postprocess.s2_decisionBoundary_masaik_plot *.clippredpkl



## 3AB. Kmeans 层次聚类
![Hierarchy](images/openlabcluster_hierarchy_plot.jpg)

准备工作。首先先给每个cluster取名字. 打开该脚本，把 **类别名称** 贴在脚本中；运行。最终，类别名称就添加到来 `*.clippredpkl`。
> python -m lilab.openlabcluster_postprocess.s3a_cluster_givename_mergefile `*.clippredpkl`


然后运行层次聚类的画图
> python -m lilab.openlabcluster_postprocess.s3b_hiecluster_plot  `*.clippredpkl`

可以合并标签
> python -m lilab.openlabcluster_postprocess.s3a2_cluster_nodemerge `*.clippredpkl`


## 3C. Mirror 或 Mutual 类别的界定
![Matrix](images/openlabcluster_cluster_mirror_mutual_matrix.jpg)
![Matrix](images/openlabcluster_cluster_mirror_mutual_notes.jpg)
判定 cluster 的Mirror 或 Mutual 关系。通过 blackFirst 和 whiteFirst 的标签对比，然后用阈值法筛选。

生成的结果在 cluster_mirror.jpg，以及在终端显示文本。

> python -m lilab.openlabcluster_postprocess.s3c_mirror_mutual_matrix_plot `*.clippredpkl`

## 4. Moseq-like Motif 的绘制
![Moseq-like](images/openlabcluster_moseq_like_motif.jpg)

需要找到原始的关键点坐标文件（都存放在`SmoothFootPkl_DIR`文件夹），才能绘制。运行下面代码。在同级目录下，会生成 `motifshowmulti` 的结果文件夹。

> python -m lilab.openlabcluster_postprocess.s4_moseq_like_motif_plot  `*.clippredpkl`   `SmoothFootPkl_DIR`

## 5. 画出每个cluster 的 3D骨架视频
![sktdraw](images/openlabcluster_cluster_sktdraw3D.jpg)

需要找到原始的关键点坐标文件（都存放在`SmoothFootPkl_DIR`文件夹），才能绘制。运行下面代码。在同级目录下，会生成 `cluster_sktdraw` 的结果文件夹。

> python -m lilab.openlabcluster_postprocess.s5_cluster_sktdraw_video  `*.clippredpkl`   `SmoothFootPkl_DIR`
