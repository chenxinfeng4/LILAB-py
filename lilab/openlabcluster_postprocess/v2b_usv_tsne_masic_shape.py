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
import cv2
import mmcv
from lilab.comm_signal.line_scale import line_scale

usv_pkl_folder = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/DayAll20230828/usv_label/SexualDevelopD35D55D75_USV_recon'
pklfile_latent = osp.join(usv_pkl_folder, 'usv_latent.usvpkl')
pklfile_evt    = osp.join(usv_pkl_folder, 'usv_evt.usvpkl')
usv_img_folder = usv_pkl_folder
nbin = 18
mosaic_size = (128, 128)
no_auto_crop = False
t_winlen = 0.26 #sec
freq_band = [20, 100]

#%%
def get_img_data(video_dir, ind_contain_xy_file):
    video_nakes = {v_i[0]:osp.join(video_dir, f'{v_i[0]}.avi')
                   for v_i in ind_contain_xy_file.ravel()
                    if v_i[0] is not None}
    video_nakes_vid = {v:mmcv.VideoReader(vfile) for v,vfile in tqdm.tqdm(video_nakes.items())}
    img_list = np.empty(ind_contain_xy_file.shape, dtype=object)
    for ix, iy in tqdm.tqdm(list(product(range(ind_contain_xy_file.shape[0]),
                                         range(ind_contain_xy_file.shape[1])))):
        video_nake, iframe = ind_contain_xy_file[ix, iy]
        if video_nake is None: continue
        vid = video_nakes_vid[video_nake]
        img = vid[iframe]
        img_mask = np.ascontiguousarray(img[:, 255:510])
        img_list[ix, iy] = img_mask
    return img_list

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

def get_cmap_white():
    # colors = ['#180239', '#2B50C3','#009600', '#DB9800', '#DB0000', '#610000']
    # # positions = [0,       0.32,     0.43,       0.66,      0.9,        1]  #比较好
    # positions = [0,       0.2,     0.4,       0.5,      0.7,        1]
    # colors = ['#0040DB', '#066F6D', '#2FA14D', '#D67921', '#971939']
    # positions = [0,         0.4,       0.5,      0.6,        1]
    
    # cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))

    # gradient = np.linspace(0, 1, 128)
    # rgb_array = cmap(gradient)[:, None, :3][::-1]

    # pil_image = Image.fromarray((rgb_array * 255).astype(np.uint8)).resize(mosaic_size)
    imgmask = '/mnt/liying.cibr.ac.cn_Xiong/decoder_layer/SexualDevelopD35D55D75_USV/tsne_usv_frequency_mask_whitebg_src.jpg'
    pil_image = Image.open(imgmask).resize(mosaic_size)
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Color(pil_image)
    # pil_image = enhancer.enhance(1.4)
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(0.8)
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

    img_contain = get_img_data(usv_img_folder, ind_contain_xy_file)

    target_res = [nbin, nbin]
    imgblack = np.zeros((*mosaic_size, 3), dtype=np.uint8)
    imblack = Image.fromarray(imgblack)
    imwhite = Image.fromarray(imgblack+255)

    immask, immask_rgb = get_cmap()
    immask_white, immask_white_rgb = get_cmap_white()
    outdir = osp.dirname(pklfile_latent)
    immask.save(osp.join(outdir, 'tsne_usv_frequency_mask.jpg'))
    immask_white.save(osp.join(outdir, 'tsne_usv_frequency_mask_whitebg.jpg'))
    from IPython.display import display

    t_wincrop_ind = [40, 256-5]
    # freq_bandcrop_ind = [20, 256-20]
    freq_bandcrop_ind = [0, 256]
    t_wincrop = 256 /  (t_wincrop_ind[1]-t_wincrop_ind[0]) * t_winlen
    freq_bandcrop = [int(i) for i in line_scale([0,256], freq_band, freq_bandcrop_ind)]
    canvas = Image.new('RGB', (mosaic_size[0]*target_res[0], mosaic_size[1]*target_res[1]))
    for ix, iy in tqdm.tqdm(list(product(range(target_res[0]),
                                         range(target_res[1])))):
        img0 = img_contain[ix, iy]
        if img0 is None:
            im = imblack
        else:
            img1 = img0 if no_auto_crop else img0[slice(*freq_bandcrop_ind), slice(*t_wincrop_ind)]
            img2 = cv2.resize(img1, mosaic_size, interpolation=cv2.INTER_AREA)
            img3 = np.clip(img2[..., [0]]/255 * 2, 0, 1) * immask_rgb
            img3[:,0] = img3[0,:] = img3[:,-1] = img3[-1,:] = 255
            im = Image.fromarray(img3.astype(np.uint8))
            # display(im)

        x, y = ix*mosaic_size[0], canvas.size[1] - iy*mosaic_size[1]
        canvas.paste(im, (x,y))
    canvas.save(osp.join(outdir, f'tsne_usv_shape_nbin{nbin}_freq{freq_bandcrop[0]}-{freq_bandcrop[1]}.png'))

    canvas = Image.new('RGB', (mosaic_size[0]*target_res[0], mosaic_size[1]*target_res[1]), color='white')
    for ix, iy in tqdm.tqdm(list(product(range(target_res[0]),
                                         range(target_res[1])))):
        img0 = img_contain[ix, iy]
        if img0 is None:
            im = imwhite
        else:
            img1 = img0 if no_auto_crop else img0[slice(*freq_bandcrop_ind), slice(*t_wincrop_ind)]
            img2 = cv2.resize(img1, mosaic_size, interpolation=cv2.INTER_AREA)
            img_rgb = img2 > 10
            img_fg = img_rgb * immask_white_rgb
            img_fg[~img_rgb] = 255
            im = Image.fromarray(img_fg)
            # print(imgfile)
            # display(im)

        x, y = ix*mosaic_size[0], canvas.size[1] - iy*mosaic_size[1]
        canvas.paste(im, (x,y))
    canvas
    canvas.save(osp.join(outdir, f'tsne_usv_shape_nbin{nbin}_freq{freq_bandcrop[0]}-{freq_bandcrop[1]}_whitebg.png'))

    return canvas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('usv_pkl_folder', type=str,
                        help='Folder containing pkl files of usv latent')
    parser.add_argument('--usv_image_folder', type=str, default=None)
    args = parser.parse_args()
    usv_pkl_folder = args.usv_pkl_folder
    pklfile_latent = osp.join(usv_pkl_folder, 'usv_latent.usvpkl')
    pklfile_evt    = osp.join(usv_pkl_folder, 'usv_evt.usvpkl')
    if args.usv_image_folder is None:
        usv_img_folder = usv_pkl_folder
    else:
        usv_img_folder = args.usv_image_folder
    show(pklfile_latent, pklfile_evt, usv_img_folder)
