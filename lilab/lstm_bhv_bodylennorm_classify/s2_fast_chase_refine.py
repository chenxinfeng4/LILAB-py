#%%
# conda activate mmdet
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import argparse


clippred_pkl  = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/24FP-behCluster/2407FP-total/out_semiseq2seq/semiseq2seq_offline.clippredpkl'


def main(clippred_pkl):
    clippred_data = pickle.load(open(clippred_pkl, 'rb'))
    project_dir = osp.dirname(clippred_pkl)
    cluster_names = clippred_data['cluster_names']
    fc = np.array([10, 24]) #start from 1. Passive, active
    sc = np.array([15, 29]) #start from 1. Passive, active
    fc_sc = np.concatenate([fc, sc])
    for i in fc_sc:
        assert 'chase' in cluster_names[i-1] or 'chasing' in cluster_names[i-1]

    feat_clips = clippred_data['feat_clips']
    speed_clips = feat_clips[:, :, :2].mean(axis=-1).mean(axis=-1)
    cluster_labels = clippred_data['cluster_labels']

    fc_chase_speed = speed_clips[np.isin(cluster_labels, fc)]
    ind_sc = np.isin(cluster_labels, sc)
    sc_chase_speed = speed_clips[ind_sc]

    #%%
    thr = 0.062   #0.058=3:1,  0.062=4:1
    fast_in_sc = sc_chase_speed > thr
    propotion = (fast_in_sc.sum() + len(fc_chase_speed)) / ((len(sc_chase_speed)) + len(fc_chase_speed)) 

    plt.figure()
    plt.subplot(121)
    plt.hist(fc_chase_speed, range=[0, 0.16], bins=40, density=True, alpha=0.5)
    plt.hist(sc_chase_speed, range=[0, 0.16], bins=40, density=True, alpha=0.5)
    plt.axvline(thr, color='r')
    plt.title(f'thr: {thr:.2f}')
    plt.subplot(122)
    plt.hist(fc_chase_speed, range=[0, 0.16], bins=40, alpha=0.5)
    plt.hist(sc_chase_speed, range=[0, 0.16], bins=40, alpha=0.5)
    plt.axvline(thr, color='r')
    plt.title(f'propotion: {propotion:.2f}')
    plt.savefig(osp.join(project_dir, 'fastchase_expend_offline.hist.pdf'))

    ind_fast_in_sc = np.where(ind_sc)[0][fast_in_sc]
    assert np.isin(cluster_labels[ind_fast_in_sc], sc).all()
    cluster_labels[ind_fast_in_sc[cluster_labels[ind_fast_in_sc]==sc[0]]] = fc[0]
    cluster_labels[ind_fast_in_sc[cluster_labels[ind_fast_in_sc]==sc[1]]] = fc[1]

    assert np.isin(cluster_labels, fc).sum() / np.isin(cluster_labels, fc_sc).sum() == propotion

    clippred_data['df_clipNames']['cluster_labels'] = cluster_labels
    clippred_data['cluster_labels'] = cluster_labels

    #%%
    outpkl = osp.join(project_dir, 'fastchase_expend_offline.clippredpkl')
    pickle.dump(clippred_data, open(outpkl, 'wb'))


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('clippred_pkl', type=str)
    args = argparse.parse_args()
    main(args.clippred_pkl)
