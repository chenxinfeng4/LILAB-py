#!/usr/bin/python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imageio
import time
import cv2
import os
import warnings
warnings.filterwarnings("ignore")
import json

from multiview_calib import utils 
from multiview_calib.bundle_adjustment_scipy import (build_input, bundle_adjustment, evaluate, 
                                                     visualisation, unpack_camera_params)
from multiview_calib.singleview_geometry import reprojection_error
from multiview_calib.extrinsics import verify_landmarks


def main(poses, landmarks, out, dump_images, setup, filenames): 
    poses = utils.json_read(poses)
    landmarks = utils.json_read(landmarks)
    views = list(landmarks.keys())
    intrinsics = {view:{'K':data['K'], 'dist':data['dist']} for view,data in poses.items()}
    extrinsics = {view:{'R':data['R'], 't':data['t']} for view,data in poses.items()}  
    camera_params, points_3d, points_2d,\
    camera_indices, point_indices, \
    n_cameras, n_points, ids, views_and_ids = build_input(views, intrinsics, extrinsics, 
                                                            landmarks, each=1, 
                                                            view_limit_triang=4)
    datadict = {'ids': np.array(ids).tolist(), 'points_3d': points_3d.tolist()}
    with open(out, 'w') as f:
        json.dump(datadict, f, indent=2)

    if dump_images:
        setup = utils.json_read(setup)
        filenames_images = utils.json_read(filenames)
        visualisation(setup, landmarks, filenames_images, 
                  camera_params, points_3d, 
                  points_2d, camera_indices, each=1, path='output/custom')  


if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')    

    parser = argparse.ArgumentParser()  
    parser.add_argument("--landmarks", "-l", type=str, required=True, default="landmarks.json",
                        help='JSON file containing the landmarks for each view')
    parser.add_argument("--poses", "-p", type=str, required=True, default="ba_poses.json",
                        help='JSON file containing the optimized poses')
    parser.add_argument("--out", "-o", type=str, required=True, default="out_points3d.json",
                        help='JSON file output for point3d')
    parser.add_argument("--filenames", "-f", type=str, required=False, default="filenames.json",
                        help='JSON file containing one filename of an image for each view. Used onyl if --dump_images is on')
    parser.add_argument("--dump_images", "-d", default=False, const=True, action='store_const',
                        help='Saves images for visualisation') 
    parser.add_argument("--setup", "-s", type=str, default=None,
                        help='setup.json') 
    args = parser.parse_args()

    main(**vars(args))
