# python -m lilab.openlabcluster_postprocess.v3a_cluster_givename_mergefile A/B/C.clippredpkl
# %%
import pickle
import os.path as osp
import argparse


cluster_names_page = """high mid trill/dots/dispersion
mid low flat/inv-U
mid long flat/trill/split
mid low short ramp/dots
mid high ramp
high mid trill
mid high ramp/jitter/split
mid jump/split
mid dots
mid high jump/trill
low shot/dots
bottom long flat/split
mid high long trill w/o step
mid ramp/inv-U
mid trill w/o step
mid flat step rill/long flat
mid dots/flat/dispersion mixed
low flat/ramp/int-U
low short/dot/jump
mid dot/dots
low long flat
mid high short trill/jump dots
mid inv-U/ramp/jitter
mid high trill w/o jump
mid low long flat/split/step trill
mid flat/inv-U
mid low dot/short
mid dot/short ramp
mid trill/jitter
mid step trill/jump dots
mid null/jump dots/split
mid jump/split/step short trill
mid slow ramp with jitter
mid dot/jump/trill/split mixed
low dot/jump/split mixed
bottom long flat/split/dispersion
bottom dot/jump/split
high dot/dots/jump
high dots/jump
mid high short trill/dots
mid long flat/jitter/trill
bottom long flat/split
mid inv-U/ramp/step short trill
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
