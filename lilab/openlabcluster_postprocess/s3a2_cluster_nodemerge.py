# python -m lilab.openlabcluster_postprocess.s3a2_cluster_nodemerge *.clippredpkl
# %%
import pickle
import numpy as np
import os.path as osp
import argparse



merge_names_page = """
2	0	15	27	32	45	23	21	Nonsocial
4	7	38	47	30	29	26	play
46	18	34	40	28	chase
14	41	25	20	36	42	39	3	6	12	Sniff
22	9	delete
1	5	8	10	11	13	16	17	19	31	35	37	43	44	rear
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
