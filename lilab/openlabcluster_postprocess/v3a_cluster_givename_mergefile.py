# python -m lilab.openlabcluster_postprocess.v3a_cluster_givename_mergefile A/B/C.clippredpkl
# %%
import pickle
import os.path as osp
import argparse


cluster_names_page = """
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
"""


def inplace_append_givename(clippredpkl_file):
    clippreddata = pickle.load(open(clippredpkl_file, "rb"))
    assert {
        "ncluster",
        "cluster_labels",
        "embedding",
        "embedding_d2",
        "clipNames",
    } <= clippreddata.keys()
    cluster_names = [s for s in cluster_names_page.split("\n") if len(s)]
    print(cluster_names)
    assert clippreddata["ncluster"] == len(cluster_names), f'{clippreddata["ncluster"]}, {len(cluster_names)}'
    clippreddata["cluster_names"] = cluster_names
    pickle.dump(clippreddata, open(clippredpkl_file, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clippredpkl", type=str)
    args = parser.parse_args()
    if osp.isfile(args.clippredpkl):
        clippredpkl = args.clippredpkl
    elif osp.isdir(args.clippredpkl):
        from lilab.openlabcluster_postprocess.s1_merge_3_file import get_assert_1_file
        clippredpkl = get_assert_1_file(osp.join(args.clippredpkl, "*.usvclippredpkl"))
    else:
        raise ValueError(f"{args.clippredpkl} is not a file or a dir")
    inplace_append_givename(clippredpkl)
