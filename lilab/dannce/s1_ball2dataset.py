# %%
import os.path as osp
import scipy.io as sio
import pickle
import numpy as np

pklfile = "/DATA/chenxinfeng/tao_rat/data/20220613-side6-addition/TPH2-KO-multiview-202201/male/cxf_batch/ball.calibpkl"
data = pickle.load(open(pklfile, "rb"))

vol_size = 160
nview = len(data["ba_poses"])

camParams = []
for i in range(nview):
    pose = data["ba_poses"][i]
    K = np.array(pose["K"]).T + [[0, 0, 0], [1, 0, 0], [1, 0, 0]]
    R = r = np.array(pose["R"]).T
    t = np.array(pose["t"])[None, :]
    dist = pose["dist"][:5]
    RDistort = np.array(dist[:2] + [dist[4]])[None, :]
    TDistort = np.array(dist[2:4])[None, :]
    camParams.append(
        {"K": K, "R": R, "r": r, "t": t, "RDistort": RDistort, "TDistort": TDistort}
    )


# %%

camnames = [f"Camera{i+1}" for i in range(nview)]
imageNames = [""]
data_3D = np.zeros((1, 14 * 3))
com3d = np.zeros((1, 3))
imageSize = np.array([data["intrinsics"][str(i)]["image_shape"] for i in range(nview)])
outdict = {
    "imageNames": imageNames,
    "camParams": camParams,
    "imageSize": imageSize,
    "data_3D": data_3D,
    "com3d": com3d,
    "vol_size": vol_size,
    "camnames": camnames,
}


outpkl = osp.splitext(pklfile)[0] + "_dummy_dannce.pkl"
pickle.dump(outdict, open(outpkl, "wb"))
sio.savemat(osp.splitext(pklfile)[0] + "_dummy_dannce.mat", outdict)
