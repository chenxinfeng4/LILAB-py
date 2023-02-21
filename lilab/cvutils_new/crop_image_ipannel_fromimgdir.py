# %%
import json
import pandas as pd
import glob
import os.path as osp
import cv2
import os
import tqdm
from lilab.cameras_setup import get_view_xywh_wrapper
# %%
pannels_xywh = get_view_xywh_wrapper(9)
outjson=glob.glob('/home/liying_lab/chenxinfeng/ftp.image_ZYQ/multiview_large/errorframe/package_errorrefine/*/*/out.json')

ipannels = [osp.basename(osp.dirname(osp.dirname(f))) for f in outjson]
ipannels = [int(iv) for iv in ipannels]
assert all(iv in range(9) for iv in ipannels)
json_this = outjson[0]

# %%
df = pd.DataFrame(columns=['video', 'iframe', 'ipannel', 'irat'])

for iv, json_this in zip(ipannels, outjson):
    json_data = json.load(open(json_this))
    for video, frames in json_data.items():
        for frame in frames:
            item={'video':video, 'iframe':frame, 'irat': None, 'ipannel':iv}
            df.loc[len(df)] = item

# %%
# set the image path
df = df.sort_values(['video', 'iframe'])

df['in_imgname'] = [osp.splitext(row['video'])[0]+'_{:06d}.jpg'.format(row['iframe'])
                    for _, row in df.iterrows()]
df['out_imgname'] = [osp.splitext(row['video'])[0]+'_pannel{}_{:06d}.jpg'.format(row['ipannel'], row['iframe'])
                    for _, row in df.iterrows()]
# %% read the image canvas files
img_dir = '/home/liying_lab/chenxinfeng/DATA/tmp_merge/outframes_raw'
files = os.listdir(img_dir)
df=df[df['in_imgname'].isin(files)]

# %% crop image canvas files -> image pannel files
out_img_dir=osp.join(img_dir,'outframes')
os.makedirs(out_img_dir, exist_ok=True)

for _, row in tqdm.tqdm(df.iterrows()):
    pth_in = osp.join(img_dir,row['in_imgname'])
    pth_out = osp.join(out_img_dir,row['out_imgname'])
    imgData = cv2.imread(pth_in)
    x,y,w,h = pannels_xywh[row['ipannel']]
    imgDataCrop = imgData[y:y+h,x:x+w]
    cv2.imwrite(pth_out, imgDataCrop, 
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
