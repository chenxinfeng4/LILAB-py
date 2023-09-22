# %%
import pickle
import os.path as osp
from sklearn.cluster import KMeans
import numpy as np
import argparse


usv_pkl_folder = '/mnt/liying.cibr.ac.cn_Xiong/USV_MP4-toXiongweiGroup/Rat_play_ZYQ_USV/Rat_play_ZYQ_USV/Rat_play_ZYQ_USV_recon'

def main(usv_pkl_folder):
    pklfile_latent = osp.join(usv_pkl_folder, 'usv_latent.usvpkl')
    pkldata_latent = pickle.load(open(pklfile_latent, 'rb'))

    data_latent = np.concatenate([pkldata_latent['usv_latent'][v]
                            for v in pkldata_latent['video_nakes']])

    kmeans = KMeans(n_clusters=20, random_state=0).fit(data_latent)
    labels = kmeans.labels_
    np.savez(osp.join(usv_pkl_folder, 'kmeans_K20.npz'), labels=labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('usv_pkl_folder', type=str)
    args = parser.parse_args()
    assert osp.isdir(args.usv_pkl_folder)
    main(args.usv_pkl_folder)
