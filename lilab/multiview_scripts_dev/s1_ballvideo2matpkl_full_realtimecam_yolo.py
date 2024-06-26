# python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_realtimecam --pannels 4 --config "E:\mmpose\res50_coco_ball_512x320_ZJF.py" --ballcalib "E:\mmpose\ball2.calibpkl"
# python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_realtimecam --pannels 9 --config "E:\mmpose\res50_coco_ball_512x320.py"
# %%
import argparse
import numpy as np
import tqdm
import torch
# from mmpose.apis import init_pose_model
import mmcv
import ffmpegcv
from torch2trt import TRTModule
import cv2
import os
from lilab.mmpose_dev.a2_convert_mmpose2engine import findcheckpoint_trt
from lilab.cameras_setup import get_view_xywh_wrapper
import itertools
from lilab.multiview_scripts_dev.s6_calibpkl_predict import CalibPredict
from torch2trt.torch2trt import torch_dtype_from_trt
from lilab.multiview_scripts_dev.comm_functions import (
      box2cs, get_max_preds, get_max_preds_gpu, transform_preds)

config_dict = {6:'/home/liying_lab/chenxinfeng/DATA/mmpose/hrnet_w32_coco_ball_512x512.py',
          10:'/home/liying_lab/chenxinfeng/DATA/mmpose/res50_coco_ball_512x512.py',
          4: r"E:\mmpose\res50_coco_ball_512x320_ZJF.py"}

preview_resize = (1280, 800)
camsize_dict = {4: (1280*2, 800*2),
                9: (1280*3, 800*3),
                10:(1280*3, 800*4)}

class DataSet():
    def __init__(self, vid, cfg, views_xywh, c_channel_in=1):
        self.vid = vid
        self.views_xywh = views_xywh
        self.c_channel_in = c_channel_in
        _, _, w, h = views_xywh[0]

        bbox = np.array([0, 0, w, h])
        center, scale = box2cs(bbox, cfg.data_cfg['image_size'])
        self.center, self.scale = center, scale

        canvas_w, canvas_h = vid.width, vid.height
        preview_resize = (canvas_w, canvas_h)
        desired_size = cfg.data_cfg['image_size'][::-1] #(h,w)
        chw_coord_ravel_nview = []
        c_channel_out=3
        assert c_channel_in==1 or c_channel_in==3
        for crop_xywh in views_xywh:
            chw_coord_ravel_nview.append(self.cv2_resize_idx_ravel((canvas_h, canvas_w), crop_xywh, desired_size, c_channel_in, c_channel_out))
        coord_NCHW_idx_ravel = np.array(chw_coord_ravel_nview)
        
        self.views_xywh = views_xywh
        self.desired_size = desired_size  #(h,w)
        self.coord_NCHW_idx_ravel = coord_NCHW_idx_ravel  #(N,C,H,W)
        self.coord_N1HW_idx_ravel = np.ascontiguousarray(coord_NCHW_idx_ravel[:,0:1,:,:])  #(N,1,H,W)
        coord_preview_CHW_idx_ravel = self.cv2_resize_idx_ravel((canvas_h, canvas_w), 
            [0, 0, canvas_w, canvas_h], [preview_resize[1], preview_resize[0]], c_channel_in, c_channel_out)
        self.coord_preview_HWC_idx_ravel = np.ascontiguousarray(coord_preview_CHW_idx_ravel.transpose(1,2,0))  #(H,W,C)

        if hasattr(self.vid, '__len__'):
            self.__len__ = self.vid.__len__

    def cv2_resize_idx_ravel(self, canvas_size, crop_xywh, desired_size, c_channel_in=1, c_channel_out=3):
        x,y,w,h = crop_xywh
        canvas_h, canvas_w = canvas_size
        h2, w2 = desired_size
        assert h/w == h2/w2
        w1_grid = np.round(np.linspace(0,w-1,w2) + x).astype(np.int64)
        h1_grid = np.round(np.linspace(0,h-1,h2) + y).astype(np.int64)
        c_grid = np.arange(c_channel_out) if c_channel_out==c_channel_in else np.array([0]*c_channel_out)
        h1_w1_c_mesh = np.meshgrid(h1_grid, w1_grid, c_grid, indexing='ij')
        hwc_coord3d = np.stack(h1_w1_c_mesh, axis=0)
        hwc_coord_ravel = np.ravel_multi_index(hwc_coord3d, [canvas_h, canvas_w, c_channel_in]).astype(np.int64)  #size = (h2, w2, c_channel_out)
        chw_coord_ravel = np.transpose(hwc_coord_ravel, (2,0,1))  #(C,H,W)
        return chw_coord_ravel

    def __iter__(self):
        vread = {1:self.vid.read_gray, 3:self.vid.read}[self.c_channel_in]
        while True:
            ret, img = vread()
            if not ret: raise StopIteration

            # img_preview = cv2.resize(img, preview_resize, interpolation=cv2.INTER_NEAREST)
            # if img_preview.ndim==2:
            #     img_preview = cv2.cvtColor(img_preview, cv2.COLOR_GRAY2BGR)
            # elif img_preview.shape[-1]==1:
            #     img_preview = np.ascontiguousarray(np.repeat(img_preview, 3, axis=-1))

            img_preview = np.zeros((preview_resize[1],preview_resize[0],3), dtype=np.uint8)
            # img_preview = img.ravel()[self.coord_preview_HWC_idx_ravel]
            img_NCHW = img.ravel()[self.coord_NCHW_idx_ravel]
            # img_N1HW = img.ravel()[self.coord_N1HW_idx_ravel]
            # img_NCHW = np.broadcast_to(img_N1HW, self.coord_NCHW_idx_ravel.shape)
            yield img_NCHW, img_preview


class MyWorker:
    def __init__(self, config, video_file, checkpoint, ballcalib, views_xywh):
        super().__init__()
        self.id = getattr(self, 'id', 0)
        self.cuda = getattr(self, 'cuda', 0)
        self.checkpoint = checkpoint
        self.calibobj = CalibPredict(ballcalib) if ballcalib else None
        camsize = camsize_dict[len(views_xywh)]
        if video_file is None:
            vid = ffmpegcv.VideoCaptureCAM("OBS Virtual Camera", 
                camsize=camsize, pix_fmt='nv12')
        else:
            assert os.path.exists(video_file)
            vid = ffmpegcv.VideoCaptureNV(video_file, pix_fmt='nv12')
        assert (vid.width, vid.height) == camsize
        print(vid.width, vid.height)
        self.video_file = video_file
        self.vid = vid
        self.views_xywh = views_xywh
        self.camsize = camsize
        cfg = mmcv.Config.fromfile(config)
        self.dataset = DataSet(self.vid, cfg, views_xywh)
        self.dataset_iter = iter(self.dataset)
        print("Well setup VideoCapture")

    def compute(self, args):
        views_xywh = self.views_xywh
        dataset, dataset_iter = self.dataset, self.dataset_iter
        center, scale = dataset.center, dataset.scale
        count_range = range(dataset.__len__()) if hasattr(dataset, '__len__') else itertools.count()
        pbar = tqdm.tqdm(count_range, desc='loading')

        with torch.cuda.device(self.cuda):
            trt_model = TRTModule()
            trt_model.load_from_engine(self.checkpoint)
            idx = trt_model.engine.get_binding_index(trt_model.input_names[0])
            input_dtype = torch_dtype_from_trt(trt_model.engine.get_binding_dtype(idx))
            input_shape = tuple(trt_model.context.get_binding_shape(idx))
            if input_shape[0]==-1:
                assert input_shape[1:]==dataset.coord_NCHW_idx_ravel.shape[1:]
                input_shape = dataset.coord_NCHW_idx_ravel.shape
            else:
                assert input_shape==dataset.coord_NCHW_idx_ravel.shape
            img_NCHW = np.zeros(input_shape)
            img_preview = np.zeros((*preview_resize,3))
            heatmap = mid_gpu(trt_model, img_NCHW, input_dtype)

            for idx, _ in enumerate(count_range, start=-1):
                pbar.update(1)
                heatmap_wait = mid_gpu(trt_model, img_NCHW, input_dtype)
                img_NCHW_next, img_preview_next = pre_cpu(dataset_iter)
                torch.cuda.current_stream().synchronize()
                img_NCHW, img_preview = img_NCHW_next, img_preview_next
                heatmap = heatmap_wait
                if idx<=-1: continue
                post_cpu(self.camsize, heatmap, center, scale, views_xywh, img_preview, self.calibobj)
                
            heatmap = mid_gpu(trt_model, img_NCHW)
            post_cpu(self.camsize, heatmap, center, scale, views_xywh, img_preview, self.calibobj)


def pre_cpu(dataset_iter):
    return next(dataset_iter)

def mid_gpu(trt_model, img_NCHW, input_dtype=torch.float32):
    batch_img = torch.from_numpy(img_NCHW).cuda().type(input_dtype)
    heatmap = trt_model(batch_img)
    return heatmap


def post_cpu(camsize, heatmap, center, scale, views_xywh, img_preview, calibobj):
    N, K, H, W = heatmap.shape
    preds, maxvals = get_max_preds_gpu(heatmap)
    preds = transform_preds(
                preds, center, scale, [W, H], use_udp=False)
    kpt_data = np.concatenate((preds, maxvals), axis=-1) #(N, xyp)
    show_kpt_data(camsize, kpt_data, views_xywh, img_preview, calibobj)


def show_kpt_data(orisize, keypoints_xyp, views_xywh, img_preview, calibobj):
    # thr
    thr = 0.4    
    indmiss = keypoints_xyp[...,2] < thr
    keypoints_xyp[indmiss] = np.nan
    keypoints_xy = keypoints_xyp[...,:2]

    # ba
    if calibobj is not None:
        keypoints_xyz_ba = calibobj.p2d_to_p3d(keypoints_xy)
        keypoints_xy_ba = calibobj.p3d_to_p2d(keypoints_xyz_ba)
    else:
        keypoints_xy_ba = keypoints_xy * np.nan

    # move
    for k1, k2, crop_xywh in zip(keypoints_xy, keypoints_xy_ba, views_xywh):
        k1[:] += np.array(crop_xywh[:2])
        k2[:] += np.array(crop_xywh[:2])

    # resize
    resize = preview_resize
    scale = (resize[0]/orisize[0], resize[1]/orisize[1])
    keypoints_xy[..., 0] *= scale[0]
    keypoints_xy[..., 1] *= scale[1]
    keypoints_xy_ba[..., 0] *= scale[0]
    keypoints_xy_ba[..., 1] *= scale[1]

    # draw
    frame = img_preview.copy()
    radius = 4
    color = [0, 0, 255]
    color_ba = [0, 255, 0]
    radius_ba = 2
    keypoints_xy = keypoints_xy.reshape(-1,2)
    keypoints_xy = keypoints_xy[~np.isnan(keypoints_xy[:,0])]
    keypoints_xy_ba = keypoints_xy_ba.reshape(-1,2)
    keypoints_xy_ba = keypoints_xy_ba[~np.isnan(keypoints_xy_ba[:,0])]
    for xy in keypoints_xy:
        cv2.circle(frame, tuple(xy.astype(np.int32)), radius, color, -1)

    for xy in keypoints_xy_ba:
        cv2.circle(frame, tuple(xy.astype(np.int32)), radius_ba, color_ba, -1)

    # imshow
    cv2.imshow('preview', frame)
    cv2.waitKey(1)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pannels', type=int, default=9, help='crop views')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--ballcalib', type=str, default=None)
    arg = parser.parse_args()

    views_xywh = get_view_xywh_wrapper(arg.pannels)
    config, checkpoint, ballcalib = arg.config, arg.checkpoint, arg.ballcalib
    if config is None:
        config = config_dict[arg.pannels]
    if checkpoint is None:
        checkpoint = findcheckpoint_trt(config, trtnake='latest.full_fp16.engine')
    print("config:", config)
    print("checkpoint:", checkpoint)

    worker = MyWorker(config, arg.video_file, checkpoint, ballcalib, views_xywh)
    worker.compute(None)
