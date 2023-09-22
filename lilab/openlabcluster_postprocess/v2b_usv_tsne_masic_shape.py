# python -m lilab.openlabcluster_postprocess.v2b_usv_tsne_masic_shape A/B
#%%
import numpy as np
from PIL import Image
import pickle
from itertools import product
import os.path as osp
from matplotlib.colors import LinearSegmentedColormap
import argparse
import tqdm

usv_pkl_folder = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/DayAll20230828/usv_label/SexualDevelopD35D55D75_USV_recon'
pklfile_latent = osp.join(usv_pkl_folder, 'usv_latent.usvpkl')
pklfile_evt    = osp.join(usv_pkl_folder, 'usv_evt.usvpkl')
usv_img_folder = usv_pkl_folder
nbin = 20
mosaic_size = (128, 128)

#%%
def get_the_centered(usv_latent, ids):
    usv_latent_ = usv_latent[ids]
    usv_latent_centered = usv_latent_ - usv_latent_.mean(axis=0, keepdims=True)
    id_center = ids[np.argmin(np.linalg.norm(usv_latent_centered, axis=1))]
    return id_center


def get_cmap():
    colors = ['#00ffff', '#dc43be','#f30005', '#fc8600', '#fffb00', '#fffd93']
    # positions = [0,       0.32,     0.43,       0.66,      0.9,        1]  #比较好
    positions = [0,       0.2,     0.43,       0.66,      0.9,        1]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))

    gradient = np.linspace(0, 1, 128)
    rgb_array = cmap(gradient)[:, None, :3][::-1]
    pil_image = Image.fromarray((rgb_array * 255).astype(np.uint8)).resize(mosaic_size)
    pil_image
    pil_image_rgb = np.array(pil_image.convert('RGB'))
    return pil_image, pil_image_rgb


def show(pklfile_latent, pklfile_evt, usv_img_folder):
    pkldata_evt    = pickle.load(open(pklfile_evt, 'rb'))
    pkldata_latent = pickle.load(open(pklfile_latent, 'rb'))

    assert tuple(pkldata_latent['video_nakes']) == tuple(pkldata_evt['video_nakes'])
    usv_latent_tsne = pkldata_latent['usv_latent_tsne']
    assert len(pkldata_evt['df_usv_evt']) == len(usv_latent_tsne)

    heat_tsne2d, xedges, yedges = pkldata_latent['heat_tsne2d__xedges__yedges']

    usv_latent = np.concatenate([pkldata_latent['usv_latent'][k]
                                for k in pkldata_latent['video_nakes']])

    xedges_re = np.linspace(xedges[0], xedges[-1], nbin+1)
    yedges_re = np.linspace(yedges[0], yedges[-1], nbin+1)
    usv_latent_tsne_cp = usv_latent_tsne.copy()
    # usv_latent_tsne_cp[usv_latent_tsne_cp[:,0]>np.mean(xedges), 0] = xedges[-1] + 100
    # usv_latent_tsne_cp[usv_latent_tsne_cp[:,1]>np.mean(yedges), 1] = yedges[-1] + 100

    ind_x = np.digitize(usv_latent_tsne_cp[:,0], xedges_re) - 1
    ind_y = np.digitize(usv_latent_tsne_cp[:,1], yedges_re) - 1

    ind_contain_xy = [[[] for _ in range(nbin)] for _ in range(nbin)]

    for id, (ix, iy) in enumerate(zip(ind_x, ind_y)):
        if ix>=nbin or iy>=nbin: continue
        ind_contain_xy[ix][iy].append(id)

    ind_contain_xy_rand = np.zeros((nbin, nbin))
    for ix, iy in product(range(nbin), range(nbin)):
        ids = ind_contain_xy[ix][iy]
        ind_contain_xy_rand[ix, iy] = get_the_centered(usv_latent, ids) if len(ids) else np.nan
#     ind_contain_xy_rand[ix, iy] = np.random.choice(ids) if len(ids) else np.nan

    ind_contain_xy_file = np.empty((nbin, nbin), dtype=object)
    for ix, iy in product(range(nbin), range(nbin)):
        idx_in_all = ind_contain_xy_rand[ix, iy]
        if np.isnan(idx_in_all):
            ind_contain_xy_file[ix, iy] = [None, None]
        else:
            ind_contain_xy_file[ix, iy] =  pkldata_evt['df_usv_evt'].iloc[int(idx_in_all)][['video_nake', 'idx_in_file']].to_list()

    target_res = [nbin, nbin]
    imgblack = np.zeros((*mosaic_size, 3), dtype=np.uint8)
    imblack = Image.fromarray(imgblack)

    immask, immask_rgb = get_cmap()
    immask.save(osp.join(usv_img_folder, 'tsne_usv_frequency_mask.jpg'))
    from IPython.display import display

    canvas = Image.new('RGB', (mosaic_size[0]*target_res[0], mosaic_size[1]*target_res[1]))
    for ix, iy in tqdm.tqdm(product(range(target_res[0]), range(target_res[1]))):
        video_nake, idx_in_file = ind_contain_xy_file[ix, iy]
        if video_nake is None:
            im = imblack
        else:
            imgfile = osp.join(usv_img_folder, video_nake, f'{idx_in_file+1:06d}.png') #start from 1
            # im1 = Image.open(imgfile).crop((100, 50, 160, 200)).resize(mosaic_size)
            im1 = Image.open(imgfile).crop((50, 50, 200, 250)).resize(mosaic_size)
            img_rgb = np.array(im1.convert('RGB')) > 10
            im = Image.fromarray(img_rgb * immask_rgb)
            # print(imgfile)
            # display(im)



        x, y = ix*mosaic_size[0], canvas.size[1] - iy*mosaic_size[1]
        canvas.paste(im, (x,y))
    canvas
    canvas.save(osp.join(usv_img_folder, f'tsne_usv_shape_nbin{nbin}.png'))
    return canvas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('usv_pkl_folder', type=str,
                        help='Folder containing pkl files of usv latent')
    args = parser.parse_args()
    usv_pkl_folder = args.usv_pkl_folder
    pklfile_latent = osp.join(usv_pkl_folder, 'usv_latent.usvpkl')
    pklfile_evt    = osp.join(usv_pkl_folder, 'usv_evt.usvpkl')
    usv_img_folder = usv_pkl_folder
    show(pklfile_latent, pklfile_evt, usv_img_folder)
