# %%
import pickle
import os
import os.path as osp
import numpy as np
import scipy.io as sio

pklfile = '/home/liying_lab/chenxinfeng/DATA/dannce/data_metric/2022-10-13Temask_anno_dannce.pkl'
outimgdir = '/home/liying_lab/chenxinfeng/DATA/dannce/data_metric/2022-10-13Te_nomask'
pkldata = pickle.load(open(pklfile,'rb'))

# %%
pklfile_out = outimgdir+'_anno_dannce.pkl'
matfile_out = osp.splitext(pklfile_out)[0] + '.mat'
assert pklfile_out!=pklfile
pkldata_out = pickle.load(open(pklfile,'rb'))
pkldata_out['imageNames'] = [osp.join(outimgdir, osp.basename(imageNames)) for imageNames in pkldata['imageNames']]
assert all(os.path.exists(imageNames) for imageNames in pkldata_out['imageNames'])

imagedict = {im:i for i, im in enumerate(pkldata_out['imageNames'])}

def getind(img):
    if 'ratblack' in img:
        ind_black = imagedict[img]
        ind_white = imagedict[img.replace('ratblack', 'ratwhite')]
    elif 'ratwhite' in img:
        ind_white = imagedict[img]
        ind_black = imagedict[img.replace('ratwhite', 'ratblack')]
    else:
        raise ValueError('img not found')
    return ind_black, ind_white

dim0, dim1 = pkldata['data_3D'].shape
data_3D = np.zeros((dim0, dim1*2))
for i, img in enumerate(pkldata_out['imageNames']):
    ind_black, ind_white = getind(img)
    data_3D[i,..., :dim1] = pkldata['data_3D'][ind_black]
    data_3D[i,..., dim1:] = pkldata['data_3D'][ind_white]
pkldata_out['data_3D'] = data_3D


pickle.dump(pkldata_out, open(pklfile_out, 'wb'))
sio.savemat(matfile_out, pkldata_out)

# %%
