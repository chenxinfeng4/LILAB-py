# python mmposepkl_2_landmarks.py E:/cxf/mmpose_rat
# %%
import json
import pickle
import sys
import os
import glob
import pandas as pd
import numpy as np

views = ["c-1", "c-2", "c-3", "c-4", "c-5", "c-6"]

likelihood_thr = 0.4

# %%
def view_link_pkl(folder_pkl):
    files_pkl = glob.glob(os.path.join(folder_pkl, "*.pkl"))
    files_pkl.sort()
    assert len(views) == len(files_pkl)
    return dict(zip(views, files_pkl))


def a1_a2_load_pkl(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
        for keypoints in data:
            assert keypoints.shape == (14, 3)
    data = np.array(data)
    x = data[:, :, 0]
    y = data[:, :, 1]
    p = data[:, :, 2]
    x_1v = x.flatten()
    y_1v = y.flatten()
    p_1v = p.flatten()
    return x_1v, y_1v, p_1v


def a3_thr_points(x_1v, y_1v, p_1v, thr=likelihood_thr):
    ids = np.arange(len(x_1v))
    idx_good = np.logical_not(np.isnan(p_1v)) & (p_1v > thr)
    px = x_1v[idx_good]
    py = y_1v[idx_good]
    ids = ids[idx_good]
    return px, py, ids


def a4_points_to_landmarksdict(px, py, ids):
    ids = ids.tolist()
    landmarks = np.array([px, py]).T.tolist()
    landmarksdict = {"ids": ids, "landmarks": landmarks}
    return landmarksdict


def read_pkl(file):
    x_1v, y_1v, p_1v = a1_a2_load_pkl(file)
    px, py, ids = a3_thr_points(x_1v, y_1v, p_1v)
    landmarksdict = a4_points_to_landmarksdict(px, py, ids)
    return landmarksdict


def main(folder_pkl, outjson):
    maps = view_link_pkl(folder_pkl)
    outdict = dict()
    for name, file in maps.items():
        print(name, os.path.split(file)[-1])
        outdict[name] = read_pkl(file)

    with open(outjson, "w") as f:
        json.dump(outdict, f, indent=4)


if __name__ == "__main__":
    n = len(sys.argv)
    if n == 1:
        folder = input("Choose the folder: >> ")
        if folder == None:
            exit()
        else:
            sys.argv.append(folder)

    print(sys.argv[1:])
    folder = sys.argv[1]
    assert sys.argv[2] in ["-o", "--out"]
    outjson = sys.argv[3]
    main(folder, outjson)
