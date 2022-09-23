# python -m lilab.multiview_scripts.rat3d_kptvideo VIDEO.mp4 black.mat white.mat
# %% imports
import numpy as np
import scipy.io as sio
import os.path as osp
import re
import json
import argparse
from lilab.multiview_scripts.rat2d_kptvideo import plot_video
from multiview_calib.singleview_geometry import project_points

pose_json = '/home/liying_lab/chenxinfeng/DATA/multiview-project/ratkeypoints_11-26/mulitview/output/global_registration/global_poses.json'

# %% functions
def load_pose(pose_json, view):
    pose_data = json.load(open(pose_json))
    params = pose_data[view]
    K = np.array(params['K'])
    R = np.array(params['R'])
    t = np.array(params['t'])
    dist = np.array(params['dist'])
    return K, R, t, dist

def load_pts3d_convert2d(mat_file, camera_params):
    K, R, t, dist = camera_params
    pts = sio.loadmat(mat_file)['points_3d']
    pts_shape = pts.shape
    pts_flattern = pts.reshape(-1, 3)
    pts2d_flattern, mask_in_front = project_points(pts_flattern, K, R, t, dist)
    pts2d = pts2d_flattern.reshape(pts_shape[:-1] + (2,))
    return pts2d

def get_view_bypattern(mat_file):
    p = re.compile(r'output_(\d+)')
    out = p.search(mat_file).group(1)
    return 'c-{}'.format(out)


def main(video, pose_json, mat_file_black=None, mat_file_white=None):
    assert osp.exists(video), 'video not found'
    assert osp.exists(pose_json), 'pose_json not found'
    assert mat_file_black or mat_file_white, 'mat_file_black or mat_file_white must be provided'
    view = get_view_bypattern(video)
    camera_params = load_pose(pose_json, view)
    if mat_file_black and mat_file_white:
        pts2d_black = load_pts3d_convert2d(mat_file_black, camera_params)
        pts2d_white = load_pts3d_convert2d(mat_file_white, camera_params)
    elif mat_file_black:
        pts2d_black = load_pts3d_convert2d(mat_file_black, camera_params)
        pts2d_white = np.zeros_like(pts2d_black) + np.nan
    elif mat_file_white:
        pts2d_white = load_pts3d_convert2d(mat_file_white, camera_params)
        pts2d_black = np.zeros_like(pts2d_white) + np.nan

    plot_video(video, pts2d_black, pts2d_white)


# %% main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the skeleton of a video')
    parser.add_argument('video', type=str, help='video file')
    parser.add_argument('mat_file_black', type=str, help='mat file of black points')
    parser.add_argument('mat_file_white', type=str, help='mat file of white points')
    args = parser.parse_args()
    main(args.video, pose_json, args.mat_file_black, args.mat_file_white)
