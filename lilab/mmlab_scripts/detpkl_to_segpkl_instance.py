# python -m lilab.mmlab_scripts.detpkl_to_segpkl_instance ./data.pkl ww_rat_800x600_merge_val_semanticLabel
# 只有1个class，但是多个 instance。根据 Ground Truth （多class 单instance） 的相似性，每个 instance 分配1个 class。
# %%
import numpy as np
import mmcv
import pycocotools.mask as maskUtils
import tqdm
import os.path as osp
import argparse

det_pkl = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/maskrcnn_seg/extract_ww_rat_800x600_0304_val_apart/data.pkl'
anno_folder = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/data/rats/ww_rat_800x600_merge_val_semanticLabel'


def get_IoU(predmasks, annomasks):
    ious = np.zeros((len(predmasks), len(annomasks)))
    for i, predmask in enumerate(predmasks):
        for j, annomask in enumerate(annomasks):
            predmask = predmask > 0
            annomask = annomask > 0
            iou = np.sum(np.logical_and(predmask, annomask)) / np.sum(np.logical_or(predmask, annomask))
            ious[i, j] = iou if np.sum(annomask) > 0 else np.nan
    return ious

def resort(anno, ious):
    anno_out = [[], []]
    for ious_anno_pred in ious.T:
        if np.isnan(ious_anno_pred).any():
            anno_out[0].append([])
            anno_out[1].append([])
        else:
            anno_out[0].append([anno[0][0][np.argmax(ious_anno_pred)]])
            anno_out[1].append([anno[1][0][np.argmax(ious_anno_pred)]])
    anno[:] = anno_out


def convert(det_pkl, anno_folder):
    pred_file_pkl = osp.dirname(det_pkl) + '/data_filename.pkl'
    imgfiles = mmcv.load(pred_file_pkl)
    pred_mask = mmcv.load(det_pkl)

    # %%
    anno_files = [osp.join(anno_folder, osp.splitext(osp.basename(imgfile))[0] + '_labelTrainIds.png') 
                    for imgfile in imgfiles]

    checklists = [osp.isfile(anno_file) for anno_file in anno_files]
    assert all(checklists)

    # %%
    annomasks = []
    predmasks = []
    nclass = 2
    for anno_file, pred_mask_i in zip(tqdm.tqdm(anno_files), pred_mask):

        annoimg = mmcv.imread(anno_file)[:, :, 0]
        annomasks = np.array([annoimg==i+2 for i in range(nclass)])

        mask_zip = pred_mask_i[1][0]
        if len(mask_zip) == 0:
            continue
        predmasks = maskUtils.decode(mask_zip)
        predmasks = predmasks.transpose((2,0,1))

        ious = get_IoU(predmasks, annomasks)
        resort(pred_mask_i, ious)

    # %%
    det_out_pkl = osp.dirname(det_pkl) + '/data_instance_as_class.pkl'
    mmcv.dump(pred_mask, det_out_pkl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('det_pkl', type=str, default=det_pkl, help='detection pkl file')
    parser.add_argument('anno_folder', type=str, default=anno_folder, help='The annotation file')
    args = parser.parse_args()
    seg_pkl = convert(args.det_pkl, args.anno_folder)
    print('seg_pkl:', seg_pkl)
