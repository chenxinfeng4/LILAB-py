# python -m lilab.multiview_scripts_dev.s5_show_calibpkl2video ../data/carliball/2022-04-29_17-58-45_ball.matcalibpkl
# %%
import pickle
import numpy as np
import ffmpegcv
import tqdm
import cv2
import os.path as osp
import argparse
from lilab.multiview_scripts_dev.s6_calibpkl_predict import CalibPredict

matfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/ball/2022-04-29_17-58-45_ball.matcalibpkl'
thr = 0.55#p value threshold

pred_colors = [[0,0,255],[233,195,120],[0,215,255]] #BGR
ba_colors = [[0,255,0],[234,100,202],[255,255,0]]
resize = (640,640)
axis_length = 220  # length of axis (mm)
key_xyz_str_format = '{:4.1f},{:4.1f},{:4.1f}'

def load_mat(matfile):
    print("Loading {}".format(matfile))
    data = pickle.load(open(matfile, 'rb'))

    views_xywh = data['views_xywh']
    
    keypoints = data['keypoints']
    indmiss = keypoints[:, :, :, 2] < thr
    keypoints_xy = keypoints[:, :, :, :2]  # (nview, times, nkeypoints, 2)
    keypoints_xy[indmiss] = np.nan
    keypoints_xy_ba = data['keypoints_xy_ba'] if len(data.get('keypoints_xy_ba', [])) else keypoints_xy+np.nan
    keypoints_xyz_ba = data['keypoints_xyz_ba'] if len(data.get('keypoints_xyz_ba', [])) else np.ones((keypoints_xy.shape[1], keypoints_xy.shape[2],3))+np.nan
    for k1, k2, crop_xywh in zip(keypoints_xy, keypoints_xy_ba, views_xywh):
        k1[:] += np.array(crop_xywh[:2])
        k2[:] += np.array(crop_xywh[:2])

    if 'ba_poses' in data:
        fun_plot_axis_line = get_axis_line(CalibPredict(data), views_xywh)
    else:
        fun_plot_axis_line = lambda x: x
    return keypoints_xy, keypoints_xy_ba, keypoints_xyz_ba, data, fun_plot_axis_line


def get_axis_line(calibPredict: CalibPredict, views_xywh:list):
    axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
    axis_recentor = np.float32([[0,0,0]])
    axis_points_recentor = axis_points + axis_recentor
    calibPredict.image_shape = None
    axis_points_recentor_xy = calibPredict.p3d_to_p2d(axis_points_recentor)     # (nview, 4, 2)
    crop_xywh_np = np.array(views_xywh)     # (nview, 4)
    axis_points_recentor_xy += crop_xywh_np[:, None, :2]
    w, h = (crop_xywh_np[:, :2] + crop_xywh_np[:, 2:]).max(axis=0)
    resize_ratio = np.array([resize[0]/w, resize[1]/h])
    axis_points_recentor_xy *= resize_ratio[None, None, :]
    axis_points_recentor_xy[np.isnan(axis_points_recentor_xy)] = 0
    axis_points_recentor_xy = axis_points_recentor_xy.astype(int)
    nview = len(crop_xywh_np)
    def plot_axis_line(img:np.ndarray):
        assert img.shape[:2][::-1] == resize
        for i in range(nview):
            image_points = axis_points_recentor_xy[i]
            origin_point = tuple(image_points[0].ravel())
            x_axis_end_point = tuple(image_points[1].ravel())  # X轴坐标
            #cv2.line(img, origin_point, x_axis_end_point, (0, 0, 255), 2)  # 绘制X轴 color red
            y_axis_end_point = tuple(image_points[2].ravel())  # Y轴坐标
            #cv2.line(img, origin_point, y_axis_end_point, (0, 255, 0), 2)  # 绘制Y轴 color green
            z_axis_end_point = tuple(image_points[3].ravel())  # Z轴坐标
            #cv2.line(img, origin_point, z_axis_end_point, (217,136, 31), 2)  # 绘制Z轴 color blue
    return plot_axis_line


def keypoint_to_video(keypoints_xy, keypoints_xy_ba, keypoints_xyz_ba, data, fun_plot_axis_line, gpu=0):
    # %%
    vfile = data['info']['vfile']
    # vin = ffmpegcv.noblock(ffmpegcv.VideoCaptureNV, vfile, resize=resize, resize_keepratio=False, gpu=gpu)
    vin = ffmpegcv.VideoCaptureNV(vfile, resize=resize, resize_keepratio=False, gpu=gpu)
    assert len(vin) <= keypoints_xy.shape[1]
    orisize = (vin.origin_width, vin.origin_height)
    scale = (resize[0]/orisize[0], resize[1]/orisize[1])

    keypoints_xy[..., 0] *= scale[0]
    keypoints_xy[..., 1] *= scale[1]
    keypoints_xy_ba[..., 0] *= scale[0]
    keypoints_xy_ba[..., 1] *= scale[1]

    def draw_point_color(frame, keypoints_xy, i, color_list, radius=3):
        keypoint_i_xy = keypoints_xy[:, i, ...] # (NVIEW,K,xy).reshape(-1, 2)
        K = keypoint_i_xy.shape[1]
        NC = len(color_list)
        color_K = [color_list[k%NC] for k in range(K)]

        for k in range(K):
            keypoint_i_xy_k = keypoint_i_xy[:, k, :]
            keypoint_i_xy_k = keypoint_i_xy_k[~np.isnan(keypoint_i_xy_k[:,0])]
            for xy in keypoint_i_xy_k:
                cv2.circle(frame, tuple(xy.astype(np.int32)), radius, color_K[k], -1)

    # %%
    # vout = ffmpegcv.noblock(ffmpegcv.VideoWriterNV,vfile.replace('.mp4', '_keypoints.mp4'), codec='h264', fps=vin.fps, gpu=gpu)
    vout = ffmpegcv.VideoWriterNV(vfile.replace('.mp4', '_keypoints.mp4'), codec='h264', fps=vin.fps, gpu=gpu)
    for i, frame in enumerate(tqdm.tqdm(vin)):
        frame = frame.copy()
        if axis_length: fun_plot_axis_line(frame)
        draw_point_color(frame, keypoints_xy, i, pred_colors, 5)
        draw_point_color(frame, keypoints_xy_ba, i, ba_colors, 3)
        if True:
            cv2.putText(frame, str(i), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
            key_xyz = keypoints_xyz_ba[i][0]
            if np.all(~np.isnan(key_xyz)):
                key_xyz_str = '{:4.1f},{:4.1f},{:4.1f}'.format(key_xyz[0], key_xyz[1], key_xyz[2])
                frame = cv2.putText(frame, key_xyz_str, (10,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        vout.write(frame)
    vout.release()


def main_showvideo(matcalibpkl, gpu=0, only3D=False):
    keypoints_xy, keypoints_xy_ba, keypoints_xyz_ba, data, fun_plot_axis_line = load_mat(matcalibpkl)
    if only3D:
        keypoints_xy[:] = np.nan
    xyz = keypoints_xyz_ba[:,0][~np.isnan(keypoints_xyz_ba[:,0])].ravel()
    if len(xyz) and np.percentile(np.abs(xyz), 95) < 10:
        global key_xyz_str_format
        key_xyz_str_format = '{:4.2f},{:4.2f},{:4.2f}'
    # refine video path
    vfile = osp.splitext(osp.abspath(matcalibpkl))[0] +\
            osp.splitext(osp.abspath(data['info']['vfile']))[1]
    data['info']['vfile'] = vfile
    print('vfile', vfile)
    keypoint_to_video(keypoints_xy, keypoints_xy_ba, keypoints_xyz_ba, data, fun_plot_axis_line, gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('matcalib', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--only3D', action='store_true')
    parser.add_argument('--axis-length', type=int, default=220)

    args = parser.parse_args()
    axis_length = args.axis_length
    main_showvideo(args.matcalib, args.gpu, args.only3D)
