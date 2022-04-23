'''
Author: your name
Date: 2021-10-03 13:47:29
LastEditTime: 2021-10-11 17:36:56
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \cxf\merge_multipart_coco.py
'''
# !pyinstaller -F merge_multipart_coco.py -i merge_multipart_coco.ico
# chenxinfeng
# ------使用方法------
# 直接拖动json到EXE中
# merge parts of 'rat1', 'rat2' into 'rat's
#


# %% import
import json
import sys, os
import datetime
from pycocotools.coco import COCO

def convert(anno, anno_out):
    coco = COCO(anno)
    dataout = dict()
    dataout['categories'] = coco.dataset['categories']
    dataout['images'] = []
    dataout['annotations'] = []
    dataout['info'] = dict(
        description = 'labelme2posecoco', 
        version = '1.0', 
        year = datetime.datetime.today().year,
        date_created = datetime.datetime.today().strftime('%Y/%m/%d'))
    dataout['licences'] = ''
    dataout['images'] = coco.dataset['images']
    # %%
    annoid =0
    imgids = coco.getImgIds()
    catids = list(coco.cats.keys())
    for image_id in imgids:
        anns = coco.imgToAnns[image_id]
        allrat_seg = {catid:[] for catid in catids}

        for iann in anns:
            catid = iann['category_id']
            allrat_seg[catid] += iann['segmentation']

        for catid, segs in allrat_seg.items():
            if segs == []: continue
            segs_1d = [j for sub in segs for j in sub]
            x_min, y_min = min(segs_1d[::2]), min(segs_1d[1::2])
            x_max, y_max = max(segs_1d[::2]), max(segs_1d[1::2])
            bbox = [int(i) for i in [x_min, y_min, x_max-x_min, y_max-y_min]]
            annotations0 = dict(
                id = annoid,
                bbox = bbox,
                iscrowd = 0,
                area = bbox[2]*bbox[3],
                category_id = catid,
                image_id = image_id,
                segmentation = segs
            )
            dataout['annotations'].append(annotations0)
            annoid+=1

    with open(anno_out, 'w') as f:
        json.dump(dataout, f, indent=4)

if __name__ == "__main__":
    assert len(sys.argv)==2
    anno = sys.argv[1]
    assert os.path.isfile(anno)
    prefix =  os.path.splitext(anno)[0]
    anno_out = prefix + '_merge.json'
    print(anno)
    convert(anno, anno_out)