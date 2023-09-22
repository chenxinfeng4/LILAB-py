# python -m lilab.openlabcluster_postprocess.s3a_cluster_givename_mergefile A/B/C.clippredpkl
# %%
import pickle
import os.path as osp
import argparse


cluster_names_page = """Facing away or far away when a rat rearing
Both rearing with contact
Mutual head tail sniff
Sniffing tail or behind
Chasing a leaving rat
Mutual head contacting parallelly
Facing toward or pressed when a rat rearing
Pinning stable
Facing away when a rat rearing
Approaching or facing toward when a rat rearing
Being leaved
Mounting, pouncing, pinning mix
Both rearing far away
Sniffing tail or behind
Rearing when being faced away
Being pinned fiercely, pounced, mounted mix
Rearing when being faced or sniffed
Rearing when being far away
Being sniffed or chased slowly
Half rearing attension
Leaving each other oppositely
Rearing when being looked or sniffed
Near each other back to back
Being tail sniffed or chased
Sniffing or chasing slowly from behind
Behind or sniffing tail when a rat rearing
Being approached
Exploring seperately back to back
Being tail sniffed
Far away when a rat rearing
Both rearing with contact
Rearing down or leaving
Crossing side by side opposite way
Approaching
Both rearing with less contact
In front of a rat and non contact
Pinning fiercely
Facing away or far away when a rat rearing
Mutual head-head contact
Chasing and sniffing when a rat rearing
Being pinned stable
Rearing when sniffed or being social range
Being chased or tail sniffed
"""


def inplace_append_givename(clippredpkl_file):
    clippreddata = pickle.load(open(clippredpkl_file, "rb"))
    assert {
        "ncluster",
        "ntwin",
        "cluster_labels",
        "embedding",
        "embedding_d2",
        "clipNames",
    } <= clippreddata.keys()
    cluster_names = [s for s in cluster_names_page.split("\n") if len(s)]
    print(cluster_names)
    assert clippreddata["ncluster"] == len(cluster_names)
    clippreddata["cluster_names"] = cluster_names
    pickle.dump(clippreddata, open(clippredpkl_file, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clippredpkl", type=str)
    args = parser.parse_args()
    assert osp.isfile(args.clippredpkl)
    inplace_append_givename(args.clippredpkl)
