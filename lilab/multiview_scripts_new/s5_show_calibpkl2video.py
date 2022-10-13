# python -m lilab.multiview_scripts_new.s5_show_calibpkl2video ../data/carliball/2022-04-29_17-58-45_ball.matcalibpkl
# %%
import pickle
import numpy as np
import ffmpegcv
import tqdm
import cv2
import os.path as osp
import argparse


matfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/ball/2022-04-29_17-58-45_ball.matcalibpkl'
thr = 0.40

def load_mat(matfile):
    print("Loading {}".format(matfile))
    data = pickle.load(open(matfile, 'rb'))

    # %%
    views_xywh = data['views_xywh']

    keypoints = data['keypoints']
    indmiss = keypoints[:, :, :, 2] < thr
    keypoints_xy = keypoints[:, :, :, :2]  # (nview, times, nkeypoints, 2)
    keypoints_xy[indmiss] = np.nan
    keypoints_xy_ba = data['keypoints_xy_ba'] if len(data.get('keypoints_xy_ba', [])) else keypoints_xy+np.nan
    keypoints_xy_baglobal = data['keypoints_xy_baglobal'] if len(data.get('keypoints_xy_baglobal', [])) else keypoints_xy+np.nan
    keypoints_xyz_baglobal = data['keypoints_xyz_baglobal'] if len(data.get('keypoints_xyz_baglobal', [])) else np.ones((keypoints_xy.shape[1], keypoints_xy.shape[2],3))+np.nan
    for k1, k2, k3, crop_xywh in zip(keypoints_xy, keypoints_xy_ba, keypoints_xy_baglobal, views_xywh):
        k1[:] += np.array(crop_xywh[:2])
        k2[:] += np.array(crop_xywh[:2])
        k3[:] += np.array(crop_xywh[:2])

    return keypoints_xy, keypoints_xy_ba, keypoints_xy_baglobal, keypoints_xyz_baglobal, data


def keypoint_to_video(keypoints_xy, keypoints_xy_ba, keypoints_xy_baglobal, keypoints_xyz_baglobal, data, gpu=0):
    # %%
    resize = (1440, 1200)

    vfile = data['info']['vfile']
    vin = ffmpegcv.VideoCaptureNV(vfile, resize=resize, resize_keepratio=False, gpu=gpu)
    assert len(vin) == keypoints_xy.shape[1]
    orisize = (vin.origin_width, vin.origin_height)
    scale = (resize[0]/orisize[0], resize[1]/orisize[1])

    keypoints_xy[..., 0] *= scale[0]
    keypoints_xy[..., 1] *= scale[1]
    keypoints_xy_ba[..., 0] *= scale[0]
    keypoints_xy_ba[..., 1] *= scale[1]
    keypoints_xy_baglobal[..., 0] *= scale[0]
    keypoints_xy_baglobal[..., 1] *= scale[1]

    def draw_point_color(frame, keypoints_xy, i, color, radius=3):
        keypoint_i_xy = keypoints_xy[:, i, :, :].reshape(-1, 2)
        keypoint_i_xy = keypoint_i_xy[~np.isnan(keypoint_i_xy[:,0])]
        # draw keypoints
        for xy in keypoint_i_xy:
            cv2.circle(frame, tuple(xy.astype(np.int32)), radius, color, -1)

    # %%
    vout = ffmpegcv.VideoWriterNV(vfile.replace('.mp4', '_keypoints.mp4'), codec='h264', fps=vin.fps, gpu=gpu)
    for i, frame in enumerate(tqdm.tqdm(vin)):
        draw_point_color(frame, keypoints_xy, i, (0,0,255), 5)
        draw_point_color(frame, keypoints_xy_ba, i, (0,255,0), 3)
        draw_point_color(frame, keypoints_xy_baglobal, i, (0,255,255), 2)
        frame = cv2.putText(frame, str(i), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        key_xyz = keypoints_xyz_baglobal[i][0]
        if np.all(~np.isnan(key_xyz)):
            key_xyz_str = '{:4.1f},{:4.1f},{:4.1f}'.format(key_xyz[0], key_xyz[1], key_xyz[2])
            frame = cv2.putText(frame, key_xyz_str, (10,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        vout.write(frame)
    vout.release()


def main_showvideo(matcalibpkl, gpu=0):
    keypoints_xy, keypoints_xy_ba, keypoints_xy_baglobal, keypoints_xyz_baglobal, data = load_mat(matcalibpkl)
    # refine video path
    vfile = data['info']['vfile']
    if not (osp.exists(vfile) and osp.isfile(vfile)):
        vfile = osp.split(osp.abspath(matcalibpkl))[0] + '/' + osp.split(osp.abspath(matcalibpkl))[1]
        data['info']['vfile'] = vfile
    keypoint_to_video(keypoints_xy, keypoints_xy_ba, keypoints_xy_baglobal, keypoints_xyz_baglobal, data, gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('matcalib', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    main_showvideo(args.matcalib, args.gpu)
