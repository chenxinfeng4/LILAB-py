# %%
import pickle
import os
import os.path as osp
import numpy as np
import scipy.io as sio

pklfile = "/home/liying_lab/chenxinfeng/DATA/dannce/data_metric/2022-10-10Tr_mask_anno_dannce.pkl"
pkldata = pickle.load(open(pklfile, "rb"))

# %%
split = 0.1
ntrial = 6

nsample = len(pkldata["data_3D"])
npick = round(nsample * split)
indpick = np.random.choice(nsample, npick, replace=False)
indpick.sort()

for i in range(ntrial):
    pklfile_out = pklfile.replace(
        "_mask_anno_dannce",
        "_trail{}_{:02d}_mask_anno_dannce".format(i + 1, int(split * 100)),
    )
    matfile_out = osp.splitext(pklfile_out)[0] + ".mat"
    assert pklfile_out != pklfile
    pkldata_out = pickle.load(open(pklfile, "rb"))
    pkldata_out["imageNames"] = [pkldata["imageNames"][ind] for ind in indpick]
    pkldata_out["data_3D"] = pkldata["data_3D"][indpick]
    pkldata_out["com3d"] = pkldata["com3d"][indpick]
    pickle.dump(pkldata_out, open(pklfile_out, "wb"))
    sio.savemat(matfile_out, pkldata_out)

# %%
