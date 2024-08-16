# %% imports

import numpy as np
import pickle
import os
import os.path

# %%
pklfile = "/home/liying_lab/chenxinfeng/DATA/dannce/demo/rat28_1280x800x9_rat_metric/DANNCE/predict_results3/latest_dannce_predict.pkl"
pkldata = pickle.load(open(pklfile, "rb"))

# %%

data_3D = []
for data_3d_bw, imageName in zip(pkldata["data_3D"], pkldata["imageNames"]):
    if "ratblack" in imageName:
        data_3D.append(data_3d_bw[:42])
    elif "ratwhite" in imageName:
        data_3D.append(data_3d_bw[42:])
    else:
        raise ValueError("Unknown camera name")
data_3D = np.array(data_3D)

pkldata["data_3D"] = data_3D

pklfileout = pklfile.replace(".pkl", "_mask.pkl")
pickle.dump(pkldata, open(pklfileout, "wb"))

# %%
