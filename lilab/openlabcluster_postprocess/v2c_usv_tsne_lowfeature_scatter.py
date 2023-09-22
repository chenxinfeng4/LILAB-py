# %%
import numpy as np
from PIL import Image
import pickle
from itertools import product
import os.path as osp
from matplotlib.colors import LinearSegmentedColormap
import argparse
import tqdm
import matplotlib.pyplot as plt
from scipy import stats
import argparse

usv_pkl_folder = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/DayAll20230828/usv_label/SexualDevelopD35D55D75_USV_recon'

def get_norm(data):
    transformed_data, lambda_ = stats.yeojohnson(data)

    # 计算转换后数据的均值和标准差
    mean = np.mean(transformed_data)
    std = np.std(transformed_data)
    normalized_data = (transformed_data - mean) / std
    normalized_data = np.clip(normalized_data, -2, 2)
    return normalized_data


def show(pklfile_latent, pklfile_evt):
    pkldata_evt    = pickle.load(open(pklfile_evt, 'rb'))
    pkldata_latent = pickle.load(open(pklfile_latent, 'rb'))
    assert tuple(pkldata_latent['video_nakes']) == tuple(pkldata_evt['video_nakes'])
    usv_latent_tsne = pkldata_latent['usv_latent_tsne']
    assert len(pkldata_evt['df_usv_evt']) == len(usv_latent_tsne)

    # %%
    df_usv_evt = pkldata_evt['df_usv_evt']
    duration = df_usv_evt['duration'].values
    main_freq = df_usv_evt['main_freq'].values
    distribution = df_usv_evt['distribution'].values
    drate = 5
    # plt.hist(get_norm(main_freq), bins=100)
    # plt.hist(main_freq, bins=100)
    plt.figure(figsize = [18, 6])
    plt.subplot(131)
    plt.scatter(usv_latent_tsne[::drate, 0], 
                usv_latent_tsne[::drate, 1],
                c = get_norm(np.log(duration[::drate]+0.001)),
                cmap = 'coolwarm')
    plt.xticks([])
    plt.yticks([])
    plt.title('Duration', fontsize=24)
    plt.gca().set_aspect('equal', 'box')

    plt.subplot(132)
    plt.scatter(usv_latent_tsne[::drate, 0], 
                usv_latent_tsne[::drate, 1],
                c = get_norm(main_freq[::drate]),
                cmap = 'coolwarm')
    plt.xticks([])
    plt.yticks([])
    plt.title('Frequency', fontsize=24)
    plt.gca().set_aspect('equal', 'box')

    plt.subplot(133)
    plt.scatter(usv_latent_tsne[::drate, 0], 
                usv_latent_tsne[::drate, 1],
                c = get_norm(distribution[::drate]),
                cmap = 'coolwarm')
    plt.xticks([])
    plt.yticks([])
    plt.title('Distribution', fontsize=24)
    plt.gca().set_aspect('equal', 'box')
    outfig = osp.join(usv_pkl_folder, 'tsne_usv_lowdimsion_feature_scatter.jpg')
    plt.savefig(outfig)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('usv_pkl_folder', type=str,
                        help='Folder containing pkl files of usv latent')
    args = parser.parse_args()
    usv_pkl_folder = args.usv_pkl_folder
    pklfile_latent = osp.join(usv_pkl_folder, 'usv_latent.usvpkl')
    pklfile_evt    = osp.join(usv_pkl_folder, 'usv_evt.usvpkl')
    show(pklfile_latent, pklfile_evt)
