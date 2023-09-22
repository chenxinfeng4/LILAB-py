# python -m lilab.openlabcluster_postprocess.v1a2_usv_latent_tsne /A/B/C/
# %%
import pickle
import os.path as osp
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from lilab.feature_autoenc.heatmap_functions import HeatmapType, heatmap
import argparse
import glob


pklfile = '/mnt/liying.cibr.ac.cn_Xiong/USV_MP4-toXiongweiGroup/Shank3_USV/usv_label/usv_latent.usvpkl'

def main(usv_pkl_folder):
    usv_pkl_files = glob.glob(osp.join(usv_pkl_folder, '*usv_latent.usvpkl'))
    assert len(usv_pkl_files) == 1
    pklfile = usv_pkl_files[0]
    pkldata = pickle.load(open(pklfile, 'rb'))
    usv_latent_dict = pkldata['usv_latent']
    usv_latent = np.concatenate([usv_latent_dict[k] for k in pkldata['video_nakes']]) #(nsample, ndim)

    # %%
    # do the t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=0)
    usv_latent_tsne = tsne.fit_transform(usv_latent)
    # plt.plot(usv_latent_tsne[:,0], usv_latent_tsne[:,1], '.')

    # %%
    heat_tsne2d,xedges,yedges = heatmap(usv_latent_tsne, bins=100)
    heatobj = HeatmapType(heat_tsne2d, xedges, yedges)
    heatobj.calculate()
    heatobj.plt_heatmap()
    heatobj.set_mask_by_densitymap(thr=0.45)
    heatobj.plt_maskheatmap()
    plt.xticks([-10,])
    plt.yticks([-10,])
    plt.xlabel('t-SNE1', fontsize=16)
    plt.ylabel('t-SNE2', fontsize=16)
    plt.title('t-SNE of USV latent', fontsize=16)
    plt.savefig(osp.join(osp.dirname(pklfile), 'usv_latent_tsne.pdf'), dpi=300, bbox_inches='tight')

    # %%
    pkldata['usv_latent_tsne'] = usv_latent_tsne
    pkldata['heat_tsne2d__xedges__yedges'] = heat_tsne2d, xedges, yedges
    pickle.dump(pkldata, open(pklfile, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('usv_pkl_folder', type=str,
                        help='Folder containing pkl files of usv latent')
    args = parser.parse_args()
    main(args.usv_pkl_folder)
