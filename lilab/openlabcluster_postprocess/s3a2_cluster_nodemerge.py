# python -m lilab.openlabcluster_postprocess.s3a2_cluster_nodemerge *.clippredpkl
# %%
import pickle
import numpy as np
import os.path as osp
import argparse



merge_names_page = """
1	Mutual contact opposite way side by side
2	Rearing up when a rat researching opposite way
3	Rearing up when being faced away
4	Sniffing tail of a rearing rat
5	Being sniffed from side
6	Sniffing tail of a rearing-up rat
7	Sniffing tail from behind
8	Facing away and turing back to a rearing rat
9	Sniffing and contact a rearing-up rat
10	Following and sniffing tail
11	Being sniffed tail side by side
12	Pouncing or pinning
13	Leaving from interaction
14	Contact beside and head up
15	Pouncing
16	Sniffing tail and approaching of a rearing rat
17	Rearing when a still rat facing away
18	Being approached
19	Being chased or pounced
20	Mutual rearing with co-attention
21	Chasing
22	Being chased
23	Mutual leaving back to back
24	Facing away in front of a rat
25	Being pounced
26	Contact same way side by side
27	Pinning
28	Sniffing neck or contact a rearing rat
29	Head contact face to face
30	Facing away back to fack
31	Being pounced
32	Being pinned
33	Rearing when being approached or sniffed tail
34	Leaving or in front of a rat
35	Mutual rearing
36	Moving opposite way side by side
37	Mutual contact head to head
38	Mutual contact and rearing
39	Rearing up when being approached or sniffed tail
40	Sniffing tail side by side
41	Sniffing and approaching of a rearing rat
42	Rearing when being sniffed tail
43	Facing away and back to a rearing-up rat
44	Approaching
45	Rearing when be sniffed back
46	Mutual rearing and facing opposite way
47	Facing away when a rat rearing
48	Rearing when a rat facing away or leaving

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
