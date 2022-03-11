# %%
import mmcv
import os
import os.path as osp
import pycocotools.mask as maskUtils
import numpy as np
import cv2 
import tqdm
import argparse
from lilab.mmlab_scripts.semantic_metrics import summery_stat

pred_file = '/home/liying_lab/chenxinfeng/DATA/mmsegmentation/seg_pkl/bw_rat_800x600_0118_cross_val/data.pkl'
anno_folder = '/home/liying_lab/chenxinfeng/DATA/mmsegmentation/data_seg/rats/annotations/test'

def calculate_matrics(pred_file, anno_folder):
    pred_file_pkl = osp.dirname(pred_file) + '/data_filename.pkl'
    imgfiles = mmcv.load(pred_file_pkl)
    pred_mask = mmcv.load(pred_file)
    # %%
    anno_files = [osp.join(anno_folder, osp.splitext(osp.basename(imgfile))[0] + '_labelTrainIds.png') 
                    for imgfile in imgfiles]

    checklists = [osp.isfile(anno_file) for anno_file in anno_files]
    assert all(checklists)

    # %%
    nomal_size = (800, 600)
    annomasks = []
    predmasks = []
    nclass = 2
    for anno_file, pred_mask_i in zip(tqdm.tqdm(anno_files), pred_mask):

        annoimg = mmcv.imread(anno_file)[:, :, 0]
        canvas = np.ones_like(annoimg)

        for iclass in range(nclass):
            if pred_mask_i[1][iclass]:
                mask_mat = maskUtils.decode(pred_mask_i[1][iclass])
                mask_mat = np.sum(mask_mat, axis=2) > 0
                canvas[mask_mat] = iclass+2

        if nomal_size != canvas.shape[::-1]:
            canvas = cv2.resize(canvas, nomal_size)
            annoimg = cv2.resize(annoimg, nomal_size)

        annomasks.append(annoimg)
        predmasks.append(canvas)

    # %%
    predmasks = np.array(predmasks)
    annomasks = np.array(annomasks)

    predmasks -= np.min(predmasks)
    annomasks -= np.min(annomasks)

    # %%
    summery_stat(predmasks, annomasks)

        
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('pred_file', type=str, default=pred_file, help='The prediction file')
    parser.add_argument('--anno_folder', type=str, default=anno_folder, help='The annotation file')
    args = parser.parse_args()
    calculate_matrics(args.pred_file, args.anno_folder)