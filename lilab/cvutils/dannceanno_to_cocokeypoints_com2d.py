# %% imports
import numpy
import os
import scipy.io as sio
import numpy as np
import os
import os.path as osp
from scipy.spatial.distance import pdist
import pickle
import argparse
from lilab.dannce.cameraIntrinsics_OpenCV import matlab_pose_to_cv2_pose

# %%
annofile = "/home/liying_lab/chenxinfeng/DATA/dannce/data/bw_rat_1280x800x9_2022-10-10_SHANK3_voxel_error/anno.mat"


def convert(annofile):
    assert osp.exists(annofile), "Annotation file does not exist"
    annodata = sio.loadmat(annofile)
    datashouldin = {"imageSize", "imageNames", "data_3D", "camParams"}
    assert datashouldin.issubset(set(annodata.keys()))
    # %%
    imageNames = [tmp[0] for tmp in np.squeeze(annodata["imageNames"])]
    dirname = osp.dirname(annofile)

    imageNames = [
        osp.join(dirname, osp.basename(imageName.replace("\\", "/")))
        for imageName in imageNames
    ]
    assert all(osp.exists(imageName) for imageName in imageNames)

    imageSize = annodata["imageSize"]  # nview x (h, w)

    # %%
    data_3D = annodata["data_3D"]  # (N, K*3)
    pts3d = data_3D.reshape((data_3D.shape[0], -1, 3))  # (N, K, 3)

    camParamsOrig = np.squeeze(annodata["camParams"])
    keys = ["K", "RDistort", "TDistort", "t", "r"]
    ba_poses = matlab_pose_to_cv2_pose(camParamsOrig)


# %%
ba_poses = matlab_pose_to_cv2_pose(camParamsOrig)
