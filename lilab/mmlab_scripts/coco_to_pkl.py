# python -m lilab.mmlab_scripts.coco_to_pkl coco.json
# %%
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import mmcv
import numpy as np
import cv2
import os
import os.path as osp
import argparse

annFile = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/data/rats/dw_rat_800x600_0304_trainval_as_bw.json'

def convert(annFile):
    coco = COCO(annFile)
    nclass =  len(coco.cats)
    classid_1based = sorted(list(coco.cats.keys()))
    classid_0based = {cid: i for i, cid in enumerate(classid_1based)}  # raw1: iclass=0

    imgIds = coco.getImgIds()
    outdata = [[[[]] * nclass,[[]] * nclass] for _ in imgIds]
    imgfiles = [coco.loadImgs(imgId)[0]['file_name'] for imgId in imgIds]
    for imgId, frame_out in zip(imgIds, outdata):
        img = coco.loadImgs(imgId)[0]
        annos = coco.imgToAnns[img['id']]
        heigh, width = img['height'], img['width']
        img_file_name = img['file_name']
        canvas = np.ones((heigh, width), dtype=np.uint8)

        for anno in annos:
            iclass = classid_0based[anno['category_id']]
            segmentations = anno['segmentation']
            polygons = []
            for segmentation in segmentations:
                polygon = np.array(segmentation).reshape((-1, 2))
                # full fill the polygon
                polygon = np.concatenate([polygon, polygon[0:1]], axis=0)
                polygon = np.round(polygon).astype(np.int32)
                polygons.append(polygon)
            cv2.fillPoly(canvas, polygons, (iclass+2))
            mask = canvas == iclass+2
            if np.sum(mask) <= 4:
                # ignore this mask
                frame_out[0][iclass] = []
            else:
                x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
                pval = 1
                frame_out[0][iclass] = [np.array([x, y, x+w, y+h, pval])]

            frame_out[1][iclass] = maskUtils.encode(
                    np.array(mask[:,:,np.newaxis], order='F', dtype=np.uint8))

    # %% write file
    datapkl = osp.join(osp.dirname(annFile), 'data.pkl')
    datafilepkl = osp.join(osp.dirname(annFile), 'data_filename.pkl')
    mmcv.dump(outdata, datapkl)
    mmcv.dump(imgfiles, datafilepkl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annFile', type=str, default=annFile, help='coco annotation file')
    args = parser.parse_args()
    convert(args.annFile)