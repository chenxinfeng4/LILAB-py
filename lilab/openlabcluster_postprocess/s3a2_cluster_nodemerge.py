# %%
import pickle
import numpy as np
import os
import os.path as osp
import re
import matplotlib.pyplot as plt
from lilab.mmpose_dev.a3_ego_align import KptEgoAligner
from matplotlib.patches import Polygon
import tqdm
from lilab.openlabcluster_postprocess.s4_moseq_like_motif_plot import parsename
import argparse
from lilab.openlabcluster_postprocess.s1_merge_3_file import get_assert_1_file


clippredpklfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day35/FWPCA0.00_P100_en3_hid30_epoch228_svm2allAcc0.93_kmeansK2use-43_fromK1-20_K100.clippredpkl'

merge_names_page = """
3	6	16	Rearing when being looked or sniffed
	19	29	Being approached or sniffed
	30	21	Chased or approached
	28	27	Facing or sniffing a rat rearing
	37	33	Both rearing
	26	23	Pinned
	8	7	Approaching
	32	9	Pinning and mounting
"""


def inplace_merge_nodes(clippredpklfile):
    merge_names = [s.strip() for s in merge_names_page.split("\n") if len(s)]
    new_clusterids = np.arange(len(merge_names)) + 100
    id_map_list = []
    for i in range(len(merge_names)):
        r = merge_names[i]
        *ids, new_label = r.split("\t")
        ids = [int(s) for s in ids]
        ids_map = [(s, new_clusterids[i], new_label) for s in ids]
        id_map_list.extend(ids_map)

    clippreddata = pickle.load(open(clippredpklfile, "rb"))
    clippreddata['cluster_nodes_merged'] = id_map_list
    pickle.dump(clippreddata, open(clippredpklfile, "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('clippredpklfile', type=str)
    args = parser.parse_args()
    assert osp.isfile(args.clippredpklfile)
    inplace_merge_nodes(args.clippredpklfile)
