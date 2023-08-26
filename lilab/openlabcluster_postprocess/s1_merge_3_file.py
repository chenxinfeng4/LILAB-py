# python -m lilab.openlabcluster_postprocess.s1_merge_3_file XXX/
# %%
import pickle
import numpy as np
import re
import argparse
import os.path as osp
import glob

# out_cluster_label = '/DATA/taoxianming/rat/result/openlabcluster/chenxinfeng/shank3Day36-2022-2023/feats35fs0.8s-overlapSign/Shank3Day36--2023-07-05/output/kmeans/FWPCA0.00_P100_en3_hid30_epoch523_svm2allAcc0.92_kmeansK2use-44_fromK1-20_K100.npz'
# in_clipNames = '/DATA/taoxianming/rat/result/openlabcluster/chenxinfeng/shank3Day36-2022-2023/feats35fs0.8s-overlapSign/clipNames.txt'
# out_embedding = '/DATA/taoxianming/rat/result/openlabcluster/chenxinfeng/shank3Day36-2022-2023/feats35fs0.8s-overlapSign/Shank3Day36--2023-07-05/output/FWPCA0.00_P100_en3_hid30_epoch523_hid_labs_simple.hlpkl'


# %%
def convert(openlabcluster_proj: str):

    if False:
        in_clipNames = osp.join(openlabcluster_proj, "videos", "clipNames.txt")
        assert osp.isfile(in_clipNames)
        clipNames = open(in_clipNames).read().splitlines()
    else:
        in_clipNames = osp.join(openlabcluster_proj, "clipNames.pkl")
        assert osp.isfile(in_clipNames)
        clipNames = pickle.load(open(in_clipNames, "rb"))['clipNames']

    out_cluster_label = get_assert_1_file(
        osp.join(
            openlabcluster_proj, "output", "*kmeans*", "*kmeansK2*euc_bio_labels.npz"
        )
    )
    merge_pklfile = osp.splitext(out_cluster_label)[0] + ".clippredpkl"
    out_cluster_label_data = np.load(out_cluster_label)
    pred_labels = out_cluster_label_data["labels"]
    assert pred_labels.min() in [0, 1]  # 从0,1开始，是Non-social
    if pred_labels.min() == 0:
        pred_labels = pred_labels + 1  # 从1开始，0被固定分做Non-social
        print("Cluster start from 1. The 0 is nonsocial not included")

    ncluster = pred_labels.max()  # 1 + ncluster = all cluster
    nsample = len(pred_labels)
    
    assert len(clipNames) == nsample

    pattern_toextract = re.compile(r"segLength(\d+)_")
    clipName = clipNames[0]
    segLength = int(pattern_toextract.findall(clipName)[0])

    out_embedding = get_assert_1_file(
        osp.join(openlabcluster_proj, "output", "kmeans", "*_simple.hlpkl")
    )
    embedding_data = pickle.load(open(out_embedding, "rb"))
    embedding_d60 = embedding_data["encFeats"]
    embedding_d2 = embedding_data["embedding"]
    assert embedding_d60.shape[0] == embedding_d2.shape[0] == nsample

    # %%
    outdata_dict = {
        "ncluster": ncluster,
        "ntwin": segLength,
        "cluster_labels": pred_labels,
        "embedding": embedding_d60,
        "embedding_d2": embedding_d2,
        "clipNames": np.array(clipNames),
    }
    pickle.dump(outdata_dict, open(merge_pklfile, "wb"))


def get_assert_1_file(globpattern: str):
    files = glob.glob(globpattern)
    files = [f for f in files if '$' not in f]
    assert len(files) == 1
    return files[0]


def convert2(cleaned_proj):
    out_embedding = get_assert_1_file(osp.join(cleaned_proj, "*_simple.hlpkl"))
    embedding_data = pickle.load(open(out_embedding, "rb"))
    assert {"encFeats", "embedding"} <= embedding_data.keys()

    out_cluster_label = get_assert_1_file(osp.join(cleaned_proj, "*svm2all*.npz"))
    out_cluster_label_data = np.load(out_cluster_label)

    in_clipNames = get_assert_1_file(osp.join(cleaned_proj, "clipNames.*"))
    if in_clipNames.endswith(".txt"):
        clipNames = open(in_clipNames).read().splitlines()
    elif in_clipNames.endswith(".pkl"):
        clipNames = pickle.load(open(in_clipNames, "rb"))['clipNames']
    else:
        raise ValueError("Unknown clipNames file")
    
    pred_labels = out_cluster_label_data["labels"]
    assert pred_labels.min() in [0, 1]  # 从0,1开始
    if pred_labels.min() == 0:
        pred_labels = pred_labels + 1  # 从1开始，0被固定分做Non-social
        print("Cluster start from 1. The 0 is nonsocial not included")

    merge_pklfile = osp.splitext(out_cluster_label)[0] + ".clippredpkl"
    ncluster = pred_labels.max()  # 1 + ncluster = all cluster
    assert (
        embedding_data["encFeats"].shape[0]
        == len(pred_labels)
        == len(out_cluster_label_data["labels"])
    )

    segLength = int(re.compile(r"segLength(\d+)_").findall(clipNames[0])[0])

    embedding_data = pickle.load(open(out_embedding, "rb"))
    embedding_d60 = embedding_data["encFeats"]
    embedding_d2 = embedding_data["embedding"]

    outdata_dict = {
        "ncluster": ncluster,
        "ntwin": segLength,
        "cluster_labels": pred_labels,
        "embedding": embedding_d60,
        "embedding_d2": embedding_d2,
        "clipNames": np.array(clipNames),
    }
    pickle.dump(outdata_dict, open(merge_pklfile, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge the files")
    parser.add_argument("openlabcluster_proj", help="Openlabcluster Project Folder")
    args = parser.parse_args()
    convert2(args.openlabcluster_proj)
