# python -m lilab.openlabcluster_postprocess.s3a2_cluster_nodemerge *.clippredpkl
# %%
import pickle
import numpy as np
import os.path as osp
import argparse


clippredpklfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/DayAll/FWPCA0.00_P100_en3_hid30_epoch350_svm2allAcc0.95_kmeansK2use-37_fromK1-20_K100.clippredpkl'

merge_names_page = """
4	8	Pin head with bitting (mirror)
5	44	In social range or sniff tail (mirror)
11	23	Approach (mirror)
15	41	Rat behind in social range (mirror)
16	32	Approach to contact (mirror)
17 	43	Pin belly with bit or kick (mirror)
20 	34	Allogroom or bit a rat's neck (mirror)
25	28	Rear when other facing away (mirror)
31	37	Leave (mirror)
33	40	Rear when other facing toward (mirror)
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
    clippreddata['cluster_nodes_mirror_merged'] = id_map_list
    pickle.dump(clippreddata, open(clippredpklfile, "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('clippredpklfile', type=str)
    args = parser.parse_args()
    assert osp.isfile(args.clippredpklfile)
    inplace_merge_nodes(args.clippredpklfile)
