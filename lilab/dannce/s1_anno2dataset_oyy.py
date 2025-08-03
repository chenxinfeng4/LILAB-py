# python -m lilab.dannce.s1_anno2dataset_oyy $datadir/anno.mat $datadir/*.mkv.calibpkl.mat $datadir/output.matcalibpkl
# %%
import argparse
import os
import os.path as osp
import pickle

import numpy as np
import scipy.io as sio
from scipy.spatial.distance import pdist

# from lilab.mmpose.s3_voxelpkl_2_matcalibpkl import matlab_pose_to_cv2_pose

# %%
annofile = "/home/liying_lab/chenxinfeng/DATA/dannce/data/simglemarmoset_2560x1440x5_20240706_voxel/anno.mat"
mat_file = "/home/liying_lab/chenxinfeng/DATA/dannce/data/simglemarmoset_2560x1440x5_20240706_voxel/2024-04-14_17-02-17.mkv.calibpkl.mat"
matcalibpkl = "/mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-4-13-singlemarmoset/test/outdir/outdir/output.matcalibpkl"
#%%
def convert(annofile, mat_file, matcalibpkl):
    pickledata = pickle.load(open(matcalibpkl, "rb"))
    kpt3d = pickledata["keypoints_xyz_ba"]
    assert osp.exists(annofile), "Annotation file does not exist"
    annodata = sio.loadmat(annofile)
    datashouldin = {"imageSize", "imageNames", "data_3D", "camParams"}
    assert datashouldin.issubset(set(annodata.keys()))
    imageNames = [tmp[0] for tmp in np.squeeze(annodata["imageNames"])]
    dirname = osp.abspath(osp.dirname(annofile))
    imageNames = [
        osp.join(dirname, osp.basename(imageName.replace("\\", "/")))
        for imageName in imageNames
    ]
    assert all(osp.exists(imageName) for imageName in imageNames)
    imageSize = annodata["imageSize"]  # nview x (h, w)

    # %%
    data_3D = annodata["data_3D"]  # (N, K*3)
    pts3d = data_3D.reshape((data_3D.shape[0], -1, 3))  # (N, K, 3)

    # com3d = (np.nanmin(pts3d, axis=1) + np.nanmax(pts3d, axis=1))/2  #(N, 3)
    com3d = np.float64(kpt3d.reshape(kpt3d.shape[0], 3))  # (N, 3)

    ind_back = 4
    com3d_back = pts3d[:, ind_back, :]  # (N, 3)
    np.putmask(com3d, ~np.isnan(com3d_back), com3d_back)
    bodylength = np.sort([np.nanmax(pdist(pts3d_now)) for pts3d_now in pts3d])
    vol_size = np.percentile(bodylength, 90)
    vol_size = np.ceil(vol_size / 10) * 10 + 20
    print("vol_size:", vol_size)

    # %%
    camParamsOrig = np.squeeze(annodata["camParams"])
    keys = ["K", "RDistort", "TDistort", "t", "r"]
    mat_data = sio.loadmat(mat_file)
    ba_poses = mat_data["ba_poses"]
    camParams = list()
    for icam in range(camParamsOrig.shape[0]):
        camParam = {key: camParamsOrig[icam][0][key][0] for key in keys}
        camParam["R"] = camParam["r"]
        camParams.append(camParam)

    # %%
    camnames = [f"Camera{i+1}" for i in range(len(camParams))]

    # %%
    outdict = {
        "imageNames": imageNames,
        "camParams": camParams,
        "ba_poses": ba_poses,
        "imageSize": imageSize,
        "data_3D": data_3D,
        "com3d": com3d,
        "vol_size": vol_size,
        "camnames": camnames,
    }
    if "pklbytes" in annodata.keys():
        outdict["pklbytes"] = annodata["pklbytes"]

    outpkl = dirname + "_anno_dannce.pkl"
    outmat = dirname + "_anno_dannce.mat"
    pickle.dump(outdict, open(outpkl, "wb"))
    outdict["imageNames"] = [
        np.array(imageName, dtype=object) for imageName in imageNames
    ]
    outdict["camnames"] = [np.array(camname, dtype=object) for camname in camnames]
    sio.savemat(outmat, outdict)
    print(outmat, outpkl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert annotation file to dannce format"
    )
    parser.add_argument("annofile", type=str, help="Annotation file")
    parser.add_argument("mat_file", type=str, help="ball calibpkl mat")
    parser.add_argument("matcalibpkl", type=str, help="640x480->com3d matcalibpkl")
    args = parser.parse_args()
    convert(args.annofile, args.mat_file, args.matcalibpkl)
