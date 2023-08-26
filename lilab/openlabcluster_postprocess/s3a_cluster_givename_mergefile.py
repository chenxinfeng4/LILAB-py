# python -m lilab.openlabcluster_postprocess.s3a_cluster_givename_mergefile A/B/C.clippredpkl
# %%
import pickle
import os.path as osp
import argparse


cluster_names_page = """Facing or sniffing a rat rearing up
Head contacting or mutual sniffing oppositly
Rearing when being faced oppositely
Exploring seperately back to back
Sniffing a rat's genital or being allogroomed on neck
Rearing when being faced other side
Approaching from behind mainly
Approching or chasing to sniffing tail or head
Pining stably
Head contact the same way
Head contact with less facing each other
Facing each other with less nose contact
Mounting or rearing down in contact or social range
Chasing
Rearing up when being sniffed, approached or faced
Rearing when being sniffed, approached or faced
Being allogroomed, pinned, mounted mixed
Both rearing with head contact or in social range
Being approached and sniffed
Rearing down in contact or in social range
Being chased
Behind or sniffing tail of a rat
Being pinned stably
Being left and approaching
Being tail sniffed and turning to contact
Being pinned fiercely and rolling
Facing or sniffing a rat rearing
Facing away when rat rearing
In front of a rat or being tail sniffed
Leaving and being approahed
Leaving each other oppositely
Pinnig,mounting mixed
Both rearing no contact
Mutual contacting mainly sniffing opposite way 
Approaching, sniffing or in social range when a rat rearing down
Mutual approaching face to face
One rearing down in both rearing in social range
Mutual walking aside oppositely with contact
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
    assert clippreddata["ncluster"] == len(cluster_names)
    clippreddata["cluster_names"] = cluster_names
    pickle.dump(clippreddata, open(clippredpkl_file, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clippredpkl", type=str)
    args = parser.parse_args()
    assert osp.isfile(args.clippredpkl)
    inplace_append_givename(args.clippredpkl)
