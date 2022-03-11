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
coco_file = '/home/liying_lab/chenxinfeng/DATA/mmpose/data/rat_keypoints/2021-11-02-bwrat_side6-kp_trainval.json'
config_yaml = '/home/liying_lab/chenxinfeng/deeplabcut-project/bwrat_28kpt-cxf-2022-02-25/config.yaml'

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
assert set(keypoints) == set([bodypart[:-1] for bodypart in bodyparts])


# copy images
coco_imgs = list(coco.imgs.values())
files = [coco_img['file_name'] for coco_img in coco_imgs]

# %% copy files for bwrat image
files_bw = [file.replace('_rat_black','').replace('_rat_white','') 
            for file in files]
files_bw_set = set(files_bw)
files_bw_setlist = list(files_bw_set)

for file in files_bw_set:
    if file[:-4] + '_rat_black.jpg' not in files:
        print(file, 'black loss')
    if file[:-4] + '_rat_white.jpg' not in files:
        print(file, 'white loss')

assert 0.95 < (len(files_bw)/2) / len(files_bw_set) <=1

f = glob.glob(osp.join(coco_dir, '*', files_bw[0]))
assert len(f) > 0, 'No image found in {}'.format(coco_dir)

img_dir = osp.dirname(f[0])
img_files = [osp.join(img_dir, img_file) for img_file in files_bw]

for img_file in img_files:
    if not osp.exists(osp.join(dlc_label_dir, osp.basename(img_file))):
        shutil.copy(img_file, dlc_label_dir)

# %% imagefile to imageid
files_to_id = {coco_img['file_name']:coco_img['id'] for coco_img in coco_imgs}


# %% merge keypoint of black and white
kpt_black, kpt_white = [], []
for file in files_bw_setlist:
    file_b = file.replace('.jpg','_rat_black.jpg')
    file_w = file.replace('.jpg','_rat_white.jpg')
    kpt_black_xy = np.ones(len(bodyparts)*2)*np.nan
    kpt_white_xy = np.ones(len(bodyparts)*2)*np.nan
    if file_b in files_to_id:
        kpt = np.array(coco.loadAnns(coco.getAnnIds(files_to_id[file_b])[0])[0]['keypoints']).reshape(-1,3)
        vis = kpt[:,2]!=2
        kpt[vis,:2] = np.nan
        kpt_xy = kpt[:,:2].flatten()
    kpt_black_xy = kpt_xy
    if file_w in files_to_id:
        kpt = np.array(coco.loadAnns(coco.getAnnIds(files_to_id[file_w])[0])[0]['keypoints']).reshape(-1,3)
        vis = kpt[:,2]!=2
        kpt[vis,:2] = np.nan
        kpt_xy = kpt[:,:2].flatten()
    kpt_white_xy = kpt_xy
    kpt_black.append(kpt_black_xy)
    kpt_white.append(kpt_white_xy)

kpt_black = np.array(kpt_black)
kpt_white = np.array(kpt_white)
kpt_bw = np.concatenate((kpt_black, kpt_white), axis=1)


# %% make up data sheet
coords = ['x', 'y']
dlc_label_dir_relative = 'labeled-data/' + coco_nake + '/'
indexfile = [dlc_label_dir_relative + imgfile for imgfile in files_bw_setlist]
header = pd.MultiIndex.from_product([[scorer], bodyparts, coords], 
                                    names=['scorer', 'bodyparts', 'coords'])


df_out = pd.DataFrame(kpt_bw, 
                      columns = header, 
                      dtype   = 'float', 
                      index   = indexfile)

# %%
h5_file_out = osp.join(dlc_label_dir,  'CollectedData_{}.h5'.format(scorer))
csv_file_out = osp.join(dlc_label_dir,  'CollectedData_{}.csv'.format(scorer))

df_out = df_out.sort_index() # sort
#df_out_repeat = df_out.loc[df_out.index.repeat(nrepeat)] # repeat



# %%
nval = 90
nfile = len(df_out)
ntrain = nfile - nval
ind_train = np.sort(np.random.choice(nfile, ntrain, replace=False))
ind_val = np.setdiff1d(np.arange(nfile), ind_train)
df_train = df_out.iloc[ind_train]
df_val = df_out.iloc[ind_val]

# 
h5_file_train = osp.join(dlc_label_dir,  'CollectedData_{}.h5'.format(scorer))
csv_file_train = osp.join(dlc_label_dir,  'CollectedData_{}.csv'.format(scorer))
df_train = df_train.loc[df_train.index.repeat(repeats=2)] # repeat
df_train.to_hdf(h5_file_train, 'df_with_missing')
df_train.to_csv(csv_file_train)

os.makedirs(osp.join(dlc_label_dir, 'val'), exist_ok=True)
h5_file_val = osp.join(dlc_label_dir,  'val/CollectedData_{}_val.h5'.format(scorer))
csv_file_val = osp.join(dlc_label_dir,  'val/CollectedData_{}_val.csv'.format(scorer))
df_val.to_hdf(h5_file_val, 'df_with_missing')
df_val.to_csv(csv_file_val)

files_val = [osp.basename(file) for file in df_val.index]
for file in files_val:
    shutil.copy(osp.join(dlc_label_dir, file), osp.join(dlc_label_dir, 'val'))
