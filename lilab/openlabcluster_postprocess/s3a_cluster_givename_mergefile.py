# python -m lilab.openlabcluster_postprocess.s3a_cluster_givename_mergefile A/B/C.clippredpkl
# %%
import pickle
import os.path as osp
import argparse


cluster_names_page = """
Rearing and being approached_ReaLow
Rearing and being sniffed tail or approached_ReaLow
Sniffing back of a rearing rat_ProLow
Slow chasing_ProHig
Mutual rearing3_CloseNon
Sniffing back of a rat_ProLow
Fast chasing_ProHig
Back to back_CloseNon
Mutual head sniffing_MutInt
Rearing and being left_CloseNon
Mutual chasing and turning around_MutInt
Being pounced from front_ReaHig
Pouncing_ProHig
Sniffing tail of a rearing rat_ProLow
Mutual approahcing face to face_MutInt
Mutual rearing face to face_CloseNon
Being pressed_ReaHig
Rearing and being approached or faced from behind_ReaLow
Approaching and sniffing tail_ProLow
Behind a rat_CloseNon
Mutual rearing with head sniff_CloseNon
Mutual sniffing tail side by side_MutInt
Following a rat_ProHig
Rearing and being sniffed_ReaLow
Pinning from side_ProHig
Rearing and a rat facing away_CloseNon
Facing away side by side _CloseNon
Being sniffed head_ReaLow
Approaching a rearing rat_ProLow
Being slow chased_ReaHig
Mutual rearing2_CloseNon
Sniffing tail and behind a rat_ProLow
Being approached_ReaLow
Facing away and a rat rearing_CloseNon
Being fast chased_ReaHig
Mutual rearing_CloseNon
Facing away and a rat rearing2_CloseNon
Being sniffed back_ReaLow
Facing away and a rat rearing3_CloseNon
Being pounced and pressed_ReaHig
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
