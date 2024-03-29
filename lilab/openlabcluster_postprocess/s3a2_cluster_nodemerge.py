# python -m lilab.openlabcluster_postprocess.s3a2_cluster_nodemerge *.clippredpkl
# %%
import pickle
import numpy as np
import os.path as osp
import argparse


clippredpklfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/DayAll/FWPCA0.00_P100_en3_hid30_epoch350_svm2allAcc0.95_kmeansK2use-37_fromK1-20_K100.clippredpkl'

merge_names_page = """
		15	18	Rearing when being faced way
	42	17	22	Rearing when being faced or sniffed
2	31	13	35	Both rearing
		8	37	Pining
		10	40	Approach or sniff when a rat rearing
	1	38	9	Facing away when a rat rearing
	28	21	23	Back to back
		5	11	Chasing or being leaved
		4	14	Sniffing tail or behind
		24	32	Leaving or being sniffed
		3	33	Mutual sniff or crossing
		16	41	Being pinned
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
