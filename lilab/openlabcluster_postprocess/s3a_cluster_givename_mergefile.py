# python -m lilab.openlabcluster_postprocess.s3a_cluster_givename_mergefile A/B/C.clippredpkl
# %%
import pickle
import os.path as osp
import argparse


cluster_names_page = """
Mutual contact opposite way side by side
Rearing up when a rat researching opposite way
Rearing up when being faced away
Sniffing tail of a rearing rat
Being sniffed from side
Sniffing tail of a rearing-up rat
Sniffing tail from behind
Facing away and turing back to a rearing rat
Sniffing and contact a rearing-up rat
Following and sniffing tail
Being sniffed tail side by side
Pouncing or pinning
Leaving from interaction
Contact beside and head up
Pouncing
Sniffing tail and approaching of a rearing rat
Rearing when a still rat facing away
Being approached
Being chased or pounced
Mutual rearing with co-attention
Chasing
Being chased
Mutual leaving back to back
Facing away in front of a rat
Being pounced
Contact same way side by side
Pinning
Sniffing neck or contact a rearing rat
Head contact face to face
Facing away back to fack
Being pounced
Being pinned
Rearing when being approached or sniffed tail
Leaving or in front of a rat
Mutual rearing
Moving opposite way side by side
Mutual contact head to head
Mutual contact and rearing
Rearing up when being approached or sniffed tail
Sniffing tail side by side
Sniffing and approaching of a rearing rat
Rearing when being sniffed tail
Facing away and back to a rearing-up rat
Approaching
Rearing when be sniffed back
Mutual rearing and facing opposite way
Facing away when a rat rearing
Rearing when a rat facing away or leaving

"""


def inplace_append_givename(clippredpkl_file, auto_label=False):
    clippreddata = pickle.load(open(clippredpkl_file, "rb"))
    assert {
        "ncluster",
        "ntwin",
        "cluster_labels",
        "embedding",
        "embedding_d2",
        "clipNames",
    } <= clippreddata.keys()
    if auto_label:
        cluster_names = [f'{s+1}' for s in range(clippreddata["ncluster"])]
    else:
        cluster_names = [s for s in cluster_names_page.split("\n") if len(s)]
    print(cluster_names)
    assert clippreddata["ncluster"] == len(cluster_names)
    clippreddata["cluster_names"] = cluster_names
    pickle.dump(clippreddata, open(clippredpkl_file, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clippredpkl", type=str)
    parser.add_argument("--auto-label",  action="store_true")
    args = parser.parse_args()
    assert osp.isfile(args.clippredpkl)
    inplace_append_givename(args.clippredpkl, args.auto_label)
