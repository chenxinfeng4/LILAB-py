# python -m lilab.openlabcluster_postprocess.v2a2_usv_merge3file $USV_DIR $NPZ
"""
usv_img_dir='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/DayAll20230828/usv_data/SexualDevelopD35D55D75_USV_recon'
kmeans_npz='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/DayAll20230828/usv_cluster/svm2allAcc0.96_kmeansK2use-30_fromK1-15.npz'
python -m lilab.openlabcluster_postprocess.v2a2_usv_merge3file $usv_img_dir $kmeans_npz
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
from PIL import Image
import tqdm
import argparse
from lilab.openlabcluster_postprocess.s1_merge_3_file import get_assert_1_file

# npz_file = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/shank3Day35-2022-2023/USV/kmeans_K1-20_labels.npz'
# npz_file = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/shank3Day35-2022-2023/USV/svm2allAcc0.9_kmeansK2use-49_fromK1-20_K100.npz'
# project_dir = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/shank3-USV_all/reconstruction'


# %%
def convert(project_dir, npz_file):
    # usv_img_dir = osp.join(project_dir, 'SexualDevelopD35D55D75_USV_recon')
    usv_img_dir = project_dir
    pklfile_evt = get_assert_1_file(osp.join(project_dir, 'usv_evt.usvpkl'))
    pklfile_latent = get_assert_1_file(osp.join(project_dir, 'usv_latent.usvpkl'))
    pklfile_merge = osp.join(project_dir, osp.basename(osp.splitext(npz_file)[0])+'.usvclippredpkl')

    npz_data = np.load(npz_file)
    usv_label = npz_data['labels'] #start from 0
    pkldata_evt = pickle.load(open(pklfile_evt, 'rb'))
    pkldata_latent = pickle.load(open(pklfile_latent, 'rb'))
    df_video_idx = pkldata_evt['df_usv_evt'][['video_nake', 'idx_in_file']]

    ncluster = usv_label.max()+1
    cluster_labels = usv_label    #从0开始
    embedding_d16 = np.concatenate([pkldata_latent['usv_latent'][f] for f in pkldata_latent['video_nakes']])
    embedding_d2  = pkldata_latent['usv_latent_tsne']
    assert len(embedding_d16) == len(embedding_d2) == len(cluster_labels) == len(df_video_idx)
    clipNames = df_video_idx.values

    outdata_dict = {
            "ncluster": ncluster,
            "isUSV": True,
            "ncluster_is_start0": True,
            "cluster_labels": cluster_labels,
            "embedding": embedding_d16,
            "embedding_d2": embedding_d2,
            "clipNames": np.array(clipNames),
    }
    pickle.dump(outdata_dict, open(pklfile_merge, "wb"))
    make_masic_example_shape(cluster_labels, ncluster, clipNames, usv_img_dir, npz_file)


def make_masic_example_shape(cluster_labels, ncluster, clipNames, usv_img_dir, npz_file):
    outdir = osp.join(osp.dirname(npz_file), f'clu_img{ncluster}_color')
    os.makedirs(outdir, exist_ok=True)

    mosaic = (8, 8)
    usv_file_all_label = []

    for i in range(ncluster):
        idx = np.where(cluster_labels==i)[0]
        naxes = mosaic[0]*mosaic[1]
        idx_choose = idx if len(idx)<naxes else np.random.choice(idx, naxes, replace=False)

        usv_file_all_label.append([osp.join(usv_img_dir, video_nake, f'{idx+1:06d}.png')
                                for video_nake, idx in clipNames[idx_choose]])

    target_size = (128, 128)
    img_maskcolor = osp.join(usv_img_dir, 'tsne_usv_frequency_mask.jpg')
    img_mask = cv2.imread(img_maskcolor)
    img_mask = cv2.resize(img_mask, target_size, interpolation=cv2.INTER_AREA)
    img_mask_rgb = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)



    for i in tqdm.trange(ncluster):
        canvas = Image.new('RGB', (mosaic[0]*target_size[0], mosaic[1]*target_size[1]))
        for img_path, (ix, iy) in zip(usv_file_all_label[i], product(range(mosaic[0]), range(mosaic[1]))):
            img0 = cv2.imread(img_path)
            img1 = img0[20:-20, 40:-5]
            img2 = cv2.resize(img1, target_size, interpolation=cv2.INTER_AREA)
            # img3 = (img2 > 10) * img_mask_rgb
            img3 = np.clip(img2[..., [0]]/255 * 2, 0, 1) * img_mask_rgb
            img3[:,0] = img3[0,:]   = 255 #= img3[-1,:]= img3[:,-1]
            im = Image.fromarray(img3.astype(np.uint8))
            canvas.paste(im, (ix*target_size[0], iy*target_size[1]))
        print(osp.join(outdir, f'cluster_{i}.png'))
        canvas.save(osp.join(outdir, f'cluster_{i}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge the files")
    parser.add_argument("usv_img_dir", help="Openlabcluster Project Folder")
    parser.add_argument("kmean_npzfile", help="npzfile")
    args = parser.parse_args()
    convert(args.usv_img_dir, args.kmean_npzfile)
    print('Done')
