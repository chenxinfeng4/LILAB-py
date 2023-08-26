# python -m lilab.openlabcluster_postprocess.s3c_mirror_mutual_matrix_plot *.clippredpkl
# %%
import pickle
import numpy as np
import os
import os.path as osp
import re
import matplotlib.pyplot as plt
from lilab.mmpose_dev.a3_ego_align import KptEgoAligner
from matplotlib.patches import Polygon
import tqdm
from lilab.openlabcluster_postprocess.s4_moseq_like_motif_plot import parsename
import argparse


clippredpklfile = '/DATA/taoxianming/rat/result/openlabcluster/chenxinfeng/shank3Day36-2022-2023/feats35fs0.8s-overlapSign/Shank3Day36--2023-07-05/output/kmeans/FWPCA0.00_P100_en3_hid30_epoch523_svm2allAcc0.92_kmeansK2use-44_fromK1-20_K100.clippredpkl'
thr = 60


def print_info(clipdata, matrix_norm):
    NCluster = clipdata['ncluster']
    cluster_names = clipdata.get('cluster_names',
                                 ['']*NCluster)
    def cluster_name_mirror(i,j):
        if cluster_names[i] and cluster_names[j]:
            outstr = f'[{cluster_names[i]}] <<<===>>> [{cluster_names[j]}]'
        else:
            outstr = ''
        return outstr
    
    def cluster_name_mutual(i):
        if cluster_names[i]:
            outstr =  f'[{cluster_names[i]}]'
        else:
            outstr = ''
        return outstr
    
    for i in range(matrix_norm.shape[0]):
        if matrix_norm[i,i] > thr:
            print('mutual', i+1, i+1, cluster_name_mutual(i))

    for i in range(matrix_norm.shape[0]):
        for j in range(i+1, matrix_norm.shape[1]):
            if matrix_norm[i,j] > thr:
                print('mirror',i+1, j+1, cluster_name_mirror(i, j))



# %%
def main(clippredpklfile):
    clipdata = pickle.load(open(clippredpklfile, 'rb'))
    cluster_labels = clipdata['cluster_labels']  #start from 1. The 0 is nonsocial

    ratblack_clip_parse = dict()
    ratwhite_clip_parse = dict()

    for cluster_label, clipName in zip(cluster_labels, clipdata['clipNames']):
        ratid = int('whiteFirst' in clipName)  #0=black, 1=white
        vname, frameid = parsename(clipName)
        if 'blackFirst' in clipName:
            ratblack_clip_parse[(vname, frameid)] = cluster_label
        else:
            ratwhite_clip_parse[(vname, frameid)] = cluster_label

    assert ratblack_clip_parse.keys() == ratwhite_clip_parse.keys()

    NCluster = clipdata['ncluster']
    matrix = np.zeros((NCluster, NCluster))
    a = np.array([(ratblack_clip_parse[k], ratwhite_clip_parse[k])
                    for k in ratblack_clip_parse.keys()])
    a -= 1 # let's start from 0
    for ai, aj in a:
        matrix[ai, aj] += 1

    # normalize
    matrix_sum0 = matrix.sum(axis=0, keepdims=True)
    matrix_sum1 = matrix.sum(axis=1, keepdims=True)
    matrix_norm0 = (matrix / matrix_sum0) * 100
    matrix_norm1 = (matrix / matrix_sum1) * 100
    matrix_norm = (matrix_norm0 + matrix_norm1)/2

    plt.figure(figsize=(20,10))
    plt.subplot(121)
    plt.imshow(matrix_norm, extent=[1-0.5, NCluster+0.5, NCluster+0.5, 1-0.5]) #start from 1
    plt.xticks(range(1, NCluster+1), rotation=45)
    plt.yticks(range(1, NCluster+1))
    plt.plot([0.5, NCluster+0.5], [0.5,NCluster+0.5], 'w')
    plt.colorbar()
    plt.xlabel('Black First labels')
    plt.ylabel('White First labels')
    plt.title('Percent')

    plt.subplot(122)
    plt.imshow(matrix_norm>60, extent=[1-0.5, NCluster+0.5, NCluster+0.5, 1-0.5])
    plt.xticks(range(1, NCluster+1), rotation=45)
    plt.yticks(range(1, NCluster+1))
    plt.plot([0.5, NCluster+0.5], [0.5,NCluster+0.5], 'w')
    for i in range(1, NCluster+1):
        plt.plot([i, i], [0.5, NCluster+0.5], 'w', linewidth=0.5)

    plt.xlabel('Black First labels')
    plt.ylabel('White First labels')
    plt.title('Percent > 60%')
    outfig = osp.join(osp.dirname(clippredpklfile), 'cluster_mirror.jpg')
    plt.savefig(outfig)

    # %%
    print_info(clipdata, matrix_norm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("clippredpkl", type=str)
    args = parser.parse_args()
    assert osp.isfile(args.clippredpkl)
    main(args.clippredpkl)
