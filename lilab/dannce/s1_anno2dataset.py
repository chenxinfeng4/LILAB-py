# %%
import scipy.io as sio
import numpy as np
import os.path as osp
from scipy.spatial.distance import pdist
import pickle
# %%
annofile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/2021-11-02-bwrat_800x600/segoutframes/2021-11-02white/anno.mat'
annodata = sio.loadmat(annofile)
datashouldin = {'imageSize', 'imageNames', 'data_3D', 'camParams'}
assert datashouldin.issubset(set(annodata.keys()))
# %%
imageNames = [tmp[0] for tmp in np.squeeze(annodata['imageNames'])]
dirname = osp.dirname(annofile)

imageNames = [osp.join(dirname, osp.basename(imageName.replace('\\', '/')))
                for imageName in imageNames]
assert all(osp.exists(imageName) for imageName in imageNames)

imageSize = annodata['imageSize']  #nview x (h, w)

# %%
data_3D = annodata['data_3D']   # (N, K*3)
pts3d = data_3D.reshape((data_3D.shape[0], -1, 3)) # (N, K, 3)

com3d = (np.nanmin(pts3d, axis=1) + np.nanmax(pts3d, axis=1))/2  #(N, 3)
bodylength = np.sort([np.nanmax(pdist(pts3d_now)) for pts3d_now in pts3d])
vol_size = np.percentile(bodylength, 90)
vol_size = np.ceil(vol_size/10 + 3) * 10

# %%
camParamsOrig = np.squeeze(annodata['camParams'])
keys = ['K', 'RDistort', 'TDistort', 't', 'r']

camParams = list()
for icam in range(camParamsOrig.shape[0]):
    camParam = {key: camParamsOrig[icam][0][key][0] for key in keys}
    camParam['R'] = camParam['r']
    camParams.append(camParam)

# %%
camnames = [f'Camera{i+1}' for i in range(len(camParams))]

# %%
outdict =  {'imageNames': imageNames,
            'camParams': camParams,
            'imageSize': imageSize,
            'data_3D': data_3D,
            'com3d': com3d,
            'vol_size': vol_size,
            'camnames': camnames}
outpkl = osp.splitext(annofile)[0] + '_dannce.pkl'
outmat = osp.splitext(annofile)[0] + '_dannce.mat'
pickle.dump(outdict, open(outpkl, 'wb'))
outdict['imageNames'] = [np.array(imageName, dtype=object)
                            for imageName in imageNames]
outdict['camnames'] = [np.array(camname, dtype=object)
                            for camname in camnames]
sio.savemat(outmat, outdict)
print(outpkl)
