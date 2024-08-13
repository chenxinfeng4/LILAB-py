# python -m lilab.openlabcluster_postprocess.v3a_cluster_givename_mergefile A/B/C.clippredpkl
# %%
import pickle
import os.path as osp
import argparse


cluster_names_page = """
Behind a rat
Chasing
Contact when both rearing
Being sniffed tail from back
Mutual crossing side by side
Behind a rearing rat
Rearing when being sniffed tail
Rearing up when being faced or sniffed
Being pounced or pinned
Facing away or leaving a rearing rat
Rearing when a rat facing away1
Rearing when a rat facing away2
Pouncing2
Pouncing
Back to back with a distance
Back to back close
Being attacked
Stay behind a rearing rat
Both rearing3
Mutual head contact
Sniffing tail of a rearing rat2
Being pinned
Mutual contact side by side2
Being pressed from back
Approaching
Being approached
Sniffing tail of a rearing rat
Mutual leaving back to back
Both rearing2
Being pounced
Both rearing1
Both rearing4
Facing away of a rearing rat2
Mutual contact side by side
Being chased
Rearing when a rat facing away3
Rearing when being approached
Facing away of a rearing rat
Mutual head contact face to face
Pinning
Being pounced or pressed
Both rearing5
Approaching or sniffing tail of a rearing rat

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
