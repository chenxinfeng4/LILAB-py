# %%
import json
import cv2
import os
import os.path as osp
import pickle
import tqdm
import numpy as np
import copy
from lilab.cvutils.auto_find_subfolderimages import find_subfolderfiles
from lilab.cameras_setup import get_view_xywh_1280x800x10 as get_view_xywh
import pycocotools.mask as maskUtils
labelme_template = {
    "version": "4.5.9",
    "flags": {},
    "shapes": [],
    "imagePath": "",
    "imageData": "",
    "imageHeight": 0,
    "imageWidth": 0,
    "lineColor": [0, 255, 0, 128],
    "fillColor": [255, 0, 0, 128],
}
# %%
def extract_frames(full_vfile, framestamps, outprefix, iview):
    datapkl = pickle.load(open(full_vfile, 'rb'))
    data = datapkl['segdata'][iview]
    crop_xywh = get_view_xywh()[iview]
    imageWidth, imageHeight = crop_xywh[2:]
    framestamps.sort()
    class_name = ['rat_black', 'rat_white']
    for frameid in framestamps:
        maskiframe = data[frameid][1]
        labelme = copy.deepcopy(labelme_template)
        labelme['imagePath'] = outprefix + '_{0:05}.jpg'.format(frameid)
        labelme['imageHeight'] = imageHeight
        labelme['imageWidth'] = imageWidth
        for maskiclass, classname in zip(maskiframe, class_name):
            if len(maskiclass)==0:
                continue
            mask = maskUtils.decode(maskiclass)
            if np.sum(mask)==0:
                continue
            # find the contours of the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                contour = contour.reshape(-1, 2)
                if len(contour)>200:
                    contour = contour[::10]
                elif len(contour)>100:
                    contour = contour[::5]
                elif len(contour)>50:
                    contour = contour[::2]
                elif len(contour)<25:
                    continue

                contour = contour.tolist()
                labelme['shapes'].append({
                    'label': classname,
                    'line_color': None,
                    'fill_color': None,
                    'points': contour,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                })
        outjson = osp.join(osp.dirname(full_vfile), osp.splitext(labelme['imagePath'])[0]+'.json')
        with open(outjson, 'w') as f:
            json.dump(labelme, f, indent=2)

# %%
jsonfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/out.json'
with open(jsonfile, 'r') as f:
    data = json.load(f)

views = get_view_xywh()
vfiles = list(data.keys())
vfile_canvas = [v[:-6]+'.segpkl' for v in vfiles]
iviews_canvas = [int(v[-5]) for v in vfiles]
full_vfile_canvas = find_subfolderfiles(jsonfile, vfile_canvas)
vfile_outprefixs = [osp.basename(osp.splitext(v)[0]) for v in vfiles]

# %%
for vfile, full_vfile, outprefix, iview in zip(tqdm.tqdm(vfiles), full_vfile_canvas, vfile_outprefixs, iviews_canvas):
    framestamps = data[vfile]
    extract_frames(full_vfile, framestamps, outprefix, iview)

# %%
