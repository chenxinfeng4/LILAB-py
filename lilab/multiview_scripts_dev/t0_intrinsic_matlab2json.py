# %%
import scipy.io as sio
import json
import numpy as np
import pickle

matfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/intrinsic.mat'
matdata = sio.loadmat(matfile)

dist = [np.squeeze(D).round(3) for D in np.squeeze(matdata['DIST_'])]
K_ = [np.squeeze(K).round(0).astype('int') for K in np.squeeze(matdata['K_'])]
assert len(dist) == len(K_)

jsondata = {}
for icam in range(len(K_)):
    jsondata[icam] = dict(
        date = "2022-10-10 21:55:50",
        description = "",
        K = K_[icam].tolist(),
        K_new = K_[icam].tolist(),
        dist = dist[icam].tolist(),
        image_shape = [800, 1280]
    )

jsonstr = json.dumps(jsondata, indent=4)
