# %%
import pickle
import glob
import numpy as np
import os.path as osp
from lilab.cameras_setup import get_json_1280x800x9

calibpkl='/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/OXTRHETxKO/ball/crop/2022-05-10_05-11ball_reverse.calibpkl'

pkldata = pickle.load(open(calibpkl, 'rb'))

assert 'ba_poses' in pkldata

outdict={}
outdict['ba_poses']={i:pkldata['ba_poses'][i] for i in range(9)}
outdict['ba_global_params']=pkldata['ba_global_params']

outdict['intrinsics']={i:pkldata['intrinsics'][str(i)] for i in range(9)}
outdict['setup'] = get_json_1280x800x9()[0]



outpkl = osp.splitext(calibpkl)[0] + '_crop9.calibpkl'
pickle.dump(outdict, open(outpkl, 'wb'))

# %%
