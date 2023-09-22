
import os.path as osp
import argparse
import lilab.openlabcluster_postprocess.s2_decisionBoundary_masaik_plot as s2

clippredpkl_file = "/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/shank3-USV_all/reconstruction/svm2allAcc0.9_kmeansK2use-49_fromK1-20_K100.usvclippredpkl"


def main(clippredpkl_file):
    outfig = osp.join(osp.dirname(clippredpkl_file), 'usv_kmeans_umap.pdf')
    s2.nbin = 100
    s2.umap_mask_threshold = 0.5
    s2.main(clippredpkl_file, outfig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('usvclippredpkl', type=str)
    args = parser.parse_args()
    assert osp.isfile(args.usvclippredpkl)
    main(args.usvclippredpkl)
