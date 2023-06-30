#python -m lilab.multiview_scripts.caliball_split_globalmove A/B/c.pkl
# %%
import pickle
import numpy as np
import argparse
import os.path as osp
import cv2
from lilab.cameras_setup import get_ballglobal_cm, get_json_1280x800x10 as get_json

global_time = [31, 424, 729, 1032, 1311 ] #sec
move_time = global_time[-1] + 3

second_based = False
matfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/ball/2022-04-29_17-58-45_ball.matpkl'


def load_mat(matfile):
    data = pickle.load(open(matfile, 'rb'))
    keypoint = data['keypoints'].copy()
    fps = data['info']['fps']
    vfile = data['info']['vfile']
    views_xywh = data['views_xywh']

    assert keypoint.ndim == 4 and keypoint.shape[2] == 1, "Only one class and one instance is supported"
    assert keypoint.shape[-1] == 3, "xyp is expected"
    keypoint = keypoint[:,:,0,:]
    return keypoint, fps, vfile, views_xywh


def split_keypoint(keypoint, fps, global_time, move_time):
    pthr = 0.71
    keypoint_p = keypoint[...,2]
    keypoint_xy = keypoint[...,:2]  # VIEWxTIMExXY
    keypoint_xy[keypoint_p < pthr] = np.nan
    if second_based:
        global_index = np.round(np.array(global_time)*fps)
        move_index = np.round(move_time*fps)
    else:
        global_index = np.array(global_time, dtype=int)
        move_index = int(move_time)
    keypoint_xy_global = keypoint_xy[:, global_index, :]      # VIEWxTIMExXY
    keypoint_xy_move = keypoint_xy[:, move_index:(-5*30), :]  # VIEWxTIMExXY
    return keypoint_xy_global, keypoint_xy_move, global_index


def get_background_img(global_iframe, vfile, views_xywh):
    background_img = None
    img_stack = []
    vin = cv2.VideoCapture(vfile)
    for i in global_iframe:
        vin.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, img = vin.read()
        assert ret, "Failed to read frame {}".format(i)
        img_stack.append(img)
    vin.release()
    img_stack = np.stack(img_stack, axis=0)
    background_canvas = np.median(img_stack, axis=0).astype(np.uint8) #HxWx3
    background_img = []
    for crop_xywh in views_xywh:
        x, y, w, h = crop_xywh
        background_img.append(background_canvas[y:y+h, x:x+w, :])
    return background_img


def convert(matfile):
    keypoint, fps, vfile, views_xywh = load_mat(matfile)
    keypoint_xy_global, keypoint_xy_move, global_index = split_keypoint(keypoint, fps, global_time, move_time)
    keypoint_xy_move_downsample = keypoint_xy_move[:,::3,:]
    background_img = get_background_img(global_index, vfile, views_xywh)
    fitball_xyz_global =  get_ballglobal_cm()
    setup_json, intrinsics_json = get_json()
    outdict = {'landmarks_global_xy': keypoint_xy_global,         # VIEWxTIMExXY
               'landmarks_move_xy': keypoint_xy_move_downsample,  # VIEWxTIMExXY
               'global_iframe': global_index,                    # TIME
               'landmarks_global_cm':  fitball_xyz_global,        # TIMExXYZ
               'background_img': background_img,                 # VIEWxHxWx3
               'setup': setup_json,                         # setup.json content
               'intrinsics': intrinsics_json,               # intrinsics.json content
               }
    outfile = osp.splitext(matfile)[0] + '.ballpkl'
    pickle.dump(outdict, open(outfile, 'wb'))
    print("{}".format(outfile))

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('matfile', type=str)
    args = parser.parse_args()
    convert(args.matfile)
    