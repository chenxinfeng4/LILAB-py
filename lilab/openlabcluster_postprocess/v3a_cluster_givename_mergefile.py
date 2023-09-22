# python -m lilab.openlabcluster_postprocess.v3a_cluster_givename_mergefile A/B/C.clippredpkl
# %%
import pickle
import os.path as osp
import argparse


cluster_names_page = """Flat or inverted-U
Trill very short 
Upward ramp trill combination
Longer very low frequency or composite
Short flat
Upward ramp or step up
Upward ramp or short
Upward ramp or long flat
Complex or short
Upward ramp or flat long
Short
Flat long
Short
Short uprward ramp
Short-longer
Upward ramp or trill combination
Trill or combination
Trill long
Upward ramp
Trill short high frequency
Low frequency or composite
High refrequnce upward ramp or inverted-U
Upward ramp or inverted-U
Trill high frequency or short
Short low frequency
Upward ramp
Flat or flat-trill combination
Trill short
Upward ramp low frequency
Upward ramp or flat
Trill or short
Flat, flat-trill combination or composite
Upward ramp or trill with jump
Trill long high frequency
Flat low frequency
Flat long or flat-trill combination
Upward ramp
Very low frequency or composite
Short low frequency
Short high frequency
Upward ramp trill combination
Trill longer short 
Composite
Trill very high frequency
Upward ramp or step up
Trill short high frequency
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
    assert osp.isfile(args.clippredpkl)
    inplace_append_givename(args.clippredpkl)
