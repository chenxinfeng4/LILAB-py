# %%
import numpy as np
import pickle
import argparse

# %%
pkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/OXTRHETxKO/ball/2022-05-09_17ball.matpkl'
data = pickle.load(open(pkl, 'rb'))

# %%
hflips = [1,3,5,6]
views = data['views_xywh']
keypoints = data['keypoints'].copy()
for iflip in hflips:
    _, _, w, _ = views[iflip]
    keypoints[iflip][..., 0] = w - keypoints[iflip][..., 0]

data['keypoints'] = keypoints

# %% 
pickle.dump(data, open(pkl, 'wb'))
print('Done!')

# %%
pkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/OXTRHETxKO/ball/2022-05-09_17ball.ballpkl'
data = pickle.load(open(pkl, 'rb'))

background_img = data['background_img']
for iflip in hflips:
    background_img[iflip] = np.fliplr(background_img[iflip])

data['background_img'] = background_img
pickle.dump(data, open(pkl, 'wb'))
print('Done!')

# %%
