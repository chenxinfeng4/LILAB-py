# python -m lilab.openlabcluster_postprocess.s3a2_cluster_nodemerge *.clippredpkl
# %%
import pickle
import numpy as np
import os.path as osp
import argparse





merge_names_page = """
1	6	10	11	12	15	16	19	28	29	31	32	33	36	38	42	CloseNon
3	5	20	23	34	39	MutInt										
2	13	14	40	ProHig												
18	21	25	27	43	ProLow											
9	17	22	30	35	41	ReaHig										
4	7	8	24	26	37	ReaLow										

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
