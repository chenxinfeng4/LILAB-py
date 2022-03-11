# %%
from pycocotools.coco import COCO
import numpy as np
import cv2
import os
import os.path as osp

annFile = '/home/liying_lab/chenxinfeng/DATA/mmsegmentation/bw_rat_800x600_0118and1130_apart_rater2.json'
coco = COCO(annFile)
out_dir_name = osp.splitext(annFile)[0] + '_semanticLabel'
os.makedirs(out_dir_name, exist_ok=True)

# %%
nclass =  len(coco.cats)
classid_1based = sorted(list(coco.cats.keys()))
classid_0based = {cid: i for i, cid in enumerate(classid_1based)}  # raw1: iclass=0

imgIds = coco.getImgIds()

for imgId in imgIds:
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

    # %% write image
    nake_file_name = osp.splitext(osp.basename(img_file_name))[0]
    out_file_name = osp.join(out_dir_name, nake_file_name + '_labelTrainIds.png')
    cv2.imwrite(out_file_name, canvas)
