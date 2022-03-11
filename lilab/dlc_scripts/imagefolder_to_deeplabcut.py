# %%imagefolder_to_deeplabcut.py
import os
import os.path as osp

imagefolder = '/home/liying_lab/chenxinfeng/deeplabcut-project/bwrat_28kpt-cxf-2022-02-25/labeled-data/2021-11-02-bwrat_side6-kp_trainval/val'

# %% format transform, imagefolder to mp4
os.chdir(imagefolder)
outfile = osp.basename(imagefolder) + '.mp4'
outfile_nake = osp.splitext(outfile)[0]

cmd_str = """ ls *.jpg *.png | tee filenames.txt | awk -v prefix="cxf" '{print "file '"'"'" $0 "'"'"'" }'  > filenames_ffmpeg.txt """ 
os.system(cmd_str)

cmd_str = f'ffmpeg -y -r 1 -f concat -i filenames_ffmpeg.txt -r 1 -c:v libx264 -y "{outfile}"'
os.system(cmd_str)

# %% deeplabcut
os.environ['DLClight'] = 'True'
import deeplabcut
config_yaml = '/home/liying_lab/chenxinfeng/deeplabcut-project/abc-edf-2021-11-23/config.yaml'
csv_postfix = deeplabcut.analyze_videos(config_yaml, [outfile], save_as_csv=True)
out_h5 = osp.splitext(outfile)[0] + csv_postfix + '.h5'

import pandas as pd
df = pd.read_hdf(out_h5, 'df_with_missing')

with open('filenames.txt', 'r') as f:
    files = [line.strip() for line in f.readlines()]

df.index = files
df = df.round(3)
df.to_csv(f'{outfile_nake}.csv')
df.to_hdf(f'{outfile_nake}.h5', 'df_with_missing')

# %% convert to coco format
coco_refer_file = '/home/liying_lab/chenxinfeng/DATA/mmpose/data/rat_keypoints/2021-11-02-bwrat_side6-kp_val.json'
from pycocotools.coco import COCO
import numpy as np
import json
coco_refer = COCO(coco_refer_file)
imgfile_to_imgid = {img['file_name']: img['id'] for img in coco_refer.imgs.values()}
assert set(files) == set(imgfile_to_imgid.keys())

df = pd.read_hdf(f'{outfile_nake}.h5', 'df_with_missing')

out_json = 'result_keypoints.json'
out_kpt = []

for file in files:
    img_id = imgfile_to_imgid[file]
    kpt = df.loc[file, :].values.round(2)
    kpt_list = kpt.tolist()
    kpt_x, kpt_y = kpt[::3], kpt[1::3]
    bbox = np.array([np.min(kpt_x), np.min(kpt_y), np.ptp(kpt_x), np.ptp(kpt_y)]).round(1).tolist()
    out_kpt.append({'image_id': img_id, 
                    'keypoints': kpt_list,
                    'bbox': bbox, 
                    'category_id': 1,
                    'score': 1.0})

# save to json
with open(out_json, 'w') as f:
    json.dump(out_kpt, f, indent=2)
