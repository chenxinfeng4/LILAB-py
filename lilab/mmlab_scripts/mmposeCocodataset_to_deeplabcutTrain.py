# %% imports
import os
import os.path as osp
import yaml
import glob
import argparse
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
import glob 
import shutil

# %% images
coco_file = '/home/liying_lab/chenxinfeng/DATA/mmpose/data/rat_keypoints/2021-11-24-wrat_side6-kp_train.json'
config_yaml = '/home/liying_lab/chenxinfeng/deeplabcut-project/abc-edf-2021-11-23/config.yaml'

nrepeat = 2

coco_nake = osp.splitext(osp.basename(coco_file))[0]
coco_dir = osp.dirname(coco_file)
dlc_dir = osp.dirname(config_yaml)
dlc_videio_dir = osp.join(dlc_dir, 'videos')
os.makedirs(dlc_videio_dir, exist_ok=True)
f =  open(dlc_videio_dir+'/'+coco_nake+'.mp4', 'a')
f.close()

dlc_label_dir = osp.join(dlc_dir, 'labeled-data', coco_nake)
os.makedirs(dlc_label_dir, exist_ok=True)


with open(config_yaml, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
bodyparts = cfg['bodyparts']
scorer = cfg['scorer']

coco = COCO(coco_file)
cats = list(coco.cats.values())[0]
keypoints = cats['keypoints']
assert keypoints == bodyparts


# copy images
coco_imgs = list(coco.imgs.values())
files = [coco_img['file_name'] for coco_img in coco_imgs]

f = glob.glob(osp.join(coco_dir, '*', files[0]))
assert len(f) > 0, 'No image found in {}'.format(coco_dir)

img_dir = osp.dirname(f[0])
img_files = [osp.join(img_dir, img_file) for img_file in files]
for img_file in img_files:
    if not osp.exists(osp.join(dlc_label_dir, osp.basename(img_file))):
        shutil.copy(img_file, dlc_label_dir)

# %% make up data sheet
coords = ['x', 'y']
dlc_label_dir_relative = 'labeled-data/' + coco_nake + '/'
indexfile = [dlc_label_dir_relative + imgfile for imgfile in files]
header = pd.MultiIndex.from_product([[scorer], bodyparts, coords], 
                                    names=['scorer', 'bodyparts', 'coords'])

bodyparts_xy_array = []
for coco_img in coco_imgs:
    bodyparts_xy = np.ones((len(bodyparts), 2)) *np.nan
    anns = coco.loadAnns(coco_img['id'])
    assert len(anns) <= 1, 'More than one annotation per image not supported'
    if len(anns) == 1:
        ann = anns[0]
        kpt = np.array(ann['keypoints']).reshape((-1, 3))
        indvis = kpt[:, 2] > 0
        bodyparts_xy[indvis, :] = kpt[indvis, :2]

    bodyparts_xy_flatten = bodyparts_xy.flatten()
    bodyparts_xy_array.append(bodyparts_xy_flatten)

bodyparts_xy_array = np.array(bodyparts_xy_array)

df_out = pd.DataFrame(bodyparts_xy_array, 
                      columns = header, 
                      dtype   = 'float', 
                      index   = indexfile)

# %%
h5_file_out = osp.join(dlc_label_dir,  'CollectedData_{}.h5'.format(scorer))
csv_file_out = osp.join(dlc_label_dir,  'CollectedData_{}.csv'.format(scorer))

df_out.sort_index() # sort
df_out_repeat = df_out.loc[df_out.index.repeat(nrepeat)] # repeat

df_out_repeat.to_hdf(h5_file_out, 'df_with_missing')
df_out_repeat.to_csv(csv_file_out)
