# python -m lilab.openlabcluster_postprocess.s3a2_cluster_nodemerge *.clippredpkl
# %%
import pickle
import numpy as np
import os.path as osp
import argparse



merge_names_page = """
0	Far Nonsocial
1	Being attacked
2	Head to head contact
3	Attack
4	Rearing  when being sniffed
5	Face to face approaching
6	Being sniffed at anogenital-1
7	Pouncing-1
8	Walking away from each other 
9	Rearing up when being faced away
10	Sniffing  anogenital
11	Being sniffed at anogenital-2
12	Being pounced
13	Sniffing a rearing rat-1
14	Circling nose to anogenital
15	Pinning
16	Both rearing and co-attention-1
17	Being chased
18	Being pinned and flipping back over-1
19	Rearing when being faced away-1
20	Being contacted
21	Being pinned and flipping back over-2
22	Being pinned and flipping back over-3
23	Both rearing and co-attention-2
24	Pouncing-2
25	Rearing when being faced away-2
26	Approaching a rearing rat
27	Contact side by side same way
28	Rearing up with a rat at the bottom
29	Facing away a rearing rat
30	Rearing when being sniffed
31	Pouncing a rat that is flipping back over 
32	At the bottom of a rearing rat
33	In the social range of a rearing rat
34	Chasing 
35	Both rearing facing away
36	Rearing when being sniffed on tha back
37	Sniffing the back of a rearing rat
38	Sniffing a rearing rat-2
39	Face to face crossing and moving away

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
