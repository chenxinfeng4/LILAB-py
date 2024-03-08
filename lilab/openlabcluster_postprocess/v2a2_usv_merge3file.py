# python -m lilab.openlabcluster_postprocess.v2a2_usv_merge3file $USV_DIR $NPZ
"""
usv_img_dir='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/DayAll20230828/usv_data/SexualDevelopD35D55D75_USV_recon'
npz_file='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/DayAll20230828/usv_cluster46/svm2allAcc0.96_kmeansK2use-30_fromK1-15.npz'
python -m lilab.openlabcluster_postprocess.v2a2_usv_merge3file $usv_img_dir $npz_file
"""
# %%
import numpy as np
import os.path as osp
import numpy as np
from PIL import Image
import pickle
from itertools import product
import cv2
import os
import tqdm
import argparse
from lilab.openlabcluster_postprocess.s1_merge_3_file import get_assert_1_file
from lilab.comm_signal.line_scale import line_scale
import mmcv
# npz_file = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/shank3Day35-2022-2023/USV/kmeans_K1-20_labels.npz'
# npz_file = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/shank3Day35-2022-2023/USV/svm2allAcc0.9_kmeansK2use-49_fromK1-20_K100.npz'
# project_dir = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/shank3-USV_all/reconstruction'
img_maskcolor = '/mnt/liying.cibr.ac.cn_Xiong/SexualDevelopD35D55D75_usv_decoder_layer__/tsne_usv_frequency_mask.jpg'
no_auto_crop = False
t_winlen = 0.26 #sec
freq_band = [20, 100]
# %%
def convert(project_dir, usv_img_dir, npz_file):
    # usv_img_dir = osp.join(project_dir, 'SexualDevelopD35D55D75_USV_recon')
    pklfile_evt = get_assert_1_file(osp.join(project_dir, 'usv_evt.usvpkl'))
    pklfile_latent = get_assert_1_file(osp.join(project_dir, 'usv_latent.usvpkl'))
    pklfile_merge = osp.join(project_dir, osp.basename(osp.splitext(npz_file)[0])+'.usvclippredpkl')

    npz_data = np.load(npz_file)
    usv_label = npz_data['labels'] #start from 0
    pkldata_evt = pickle.load(open(pklfile_evt, 'rb'))
    pkldata_latent = pickle.load(open(pklfile_latent, 'rb'))
    df_video_idx = pkldata_evt['df_usv_evt'][['video_nake', 'idx_in_file']]
    df_usv_evt = pkldata_evt['df_usv_evt']
    df_usv_evt['cluster_labels'] = usv_label

    ncluster = usv_label.max()+1
    cluster_labels = usv_label    #从0开始
    embedding_d16 = np.concatenate([pkldata_latent['usv_latent'][f] for f in pkldata_latent['video_nakes']])
    embedding_d2  = pkldata_latent.get('usv_latent_tsne', np.zeros((embedding_d16.shape[0], 2))+np.nan)
    assert len(embedding_d16) == len(embedding_d2) == len(cluster_labels) == len(df_video_idx)
    clipNames = df_video_idx.values

    outdata_dict = {
            "ncluster": ncluster,
            "isUSV": True,
            "ncluster_is_start0": True,
            "cluster_labels": cluster_labels,
            'df_usv_evt': df_usv_evt,
            "embedding": embedding_d16,
            "embedding_d2": embedding_d2,
            "clipNames": np.array(clipNames),

    }
    pickle.dump(outdata_dict, open(pklfile_merge, "wb"))
    make_masic_example_shape(project_dir, cluster_labels, ncluster, clipNames, usv_img_dir)


def make_masic_example_shape(project_dir, cluster_labels, ncluster, clipNames, usv_img_dir):
    mosaic = (8, 8)
    video_nakes = set([clipName[0] for clipName in clipNames])
    # video_nakes_vid = {v: mmcv.VideoReader(osp.join(usv_img_dir, v+'.avi'))
    #                    for v in tqdm.tqdm(video_nakes, desc='Init videos')}
    video_nakes_vid = {v: mmcv.VideoReader(osp.join(usv_img_dir, v+'_m.avi'))
                       for v in tqdm.tqdm(video_nakes, desc='Init videos')}
    img_list = np.empty(ncluster, dtype=object)
    for i in tqdm.trange(ncluster, desc='Read frames'):
        idx = np.where(cluster_labels==i)[0]
        naxes = mosaic[0]*mosaic[1]
        idx_choose = idx if len(idx)<naxes else np.random.choice(idx, naxes, replace=False)
        clip_choose = clipNames[idx_choose]
        # img_list[i] = [np.ascontiguousarray(video_nakes_vid[v][iframe][:, 255:510])
        #                 for v, iframe in clip_choose]
        img_list[i] = [np.ascontiguousarray(video_nakes_vid[v][iframe])
                        for v, iframe in clip_choose]

    target_size = (128, 128)
    assert osp.isfile(img_maskcolor)
    img_mask = cv2.imread(img_maskcolor)
    img_mask = cv2.resize(img_mask, target_size, interpolation=cv2.INTER_AREA)
    img_mask_rgb = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)

    t_wincrop_ind = [40, 256-5]
    # freq_bandcrop_ind = [20, 256-20]
    freq_bandcrop_ind = [0, 256-20]
    t_wincrop = 256 /  (t_wincrop_ind[1]-t_wincrop_ind[0]) * t_winlen
    freq_bandcrop = [int(i) for i in line_scale([0,256], freq_band, freq_bandcrop_ind)]
    if no_auto_crop: #no crop
        dirname =  f'clu_img{ncluster}_color'
    else: #crop
        dirname =  f'clu_img{ncluster}_t{t_wincrop:.2f}_freq{freq_bandcrop[0]}-{freq_bandcrop[1]}_color'
    outdir = osp.join(project_dir, dirname)
    os.makedirs(outdir, exist_ok=True)

    for i in range(ncluster):
        canvas = Image.new('RGB', (mosaic[0]*target_size[0], mosaic[1]*target_size[1]))
        for img0, (ix, iy) in zip(img_list[i], product(range(mosaic[0]), range(mosaic[1]))):
            img1 = img0 if no_auto_crop else img0[slice(*freq_bandcrop_ind), slice(*t_wincrop_ind)]
            img2 = cv2.resize(img1, target_size, interpolation=cv2.INTER_AREA)
            img3 = np.clip(img2[..., [0]]/255 * 2, 0, 1) * img_mask_rgb
            img3[:,0] = img3[0,:] = 255 #= img3[-1,:]= img3[:,-1]
            im = Image.fromarray(img3.astype(np.uint8))
            canvas.paste(im, (ix*target_size[0], iy*target_size[1]))
        print(osp.join(outdir, f'cluster_{i}.png'))
        canvas.save(osp.join(outdir, f'cluster_{i}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge the files")

    parser.add_argument("usv_pkl_dir", help="Openlabcluster Project Folder")
    parser.add_argument("kmean_npzfile", help="npzfile")
    parser.add_argument("--usv_img_dir", default=None, type=str)
    parser.add_argument("--no-auto-crop", action="store_true", default=False)
    args = parser.parse_args()
    no_auto_crop = args.no_auto_crop
    if args.usv_img_dir is None:
        args.usv_img_dir = args.usv_pkl_dir
    convert(args.usv_pkl_dir, args.usv_img_dir, args.kmean_npzfile)
    print('Done')
