# python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_realtimecam_copy2 --pannels 4 --config "G:\mmpose\res50_coco_ball_512x320_ZJF.py" --ballcalib "G:\mmpose\ball2.calibpkl"
# %%
import argparse
import os
import re
import os.path as osp
import glob
import numpy as np
import tqdm
import torch
from mmpose.apis import init_pose_model
import ffmpegcv
from torch2trt import TRTModule
import itertools
import cv2
from lilab.mmpose_dev.a2_convert_mmpose2trt import findcheckpoint_trt
from lilab.cameras_setup import get_view_xywh_wrapper
from functools import partial

from threading import Thread
from queue import Queue
import multiprocessing
from lilab.multiview_scripts_dev.s6_calibpkl_predict import CalibPredict
# from multiprocessing import Process as Thread
# from multiprocessing import Queue


config_dict = {6:'/home/liying_lab/chenxinfeng/DATA/mmpose/hrnet_w32_coco_ball_512x512.py',
          10:'/home/liying_lab/chenxinfeng/DATA/mmpose/res50_coco_ball_512x512.py',
          4: '/home/liying_lab/chenxinfeng/DATA/mmpose/res50_coco_ball_512x512_ZJF.py'}

pos_views = []
preview_resize = (1280, 800)
camsize = [2560, 1600]

q = Queue(maxsize=1)

class ProducerThread(Thread):
    def __init__(self, host):
        super(ProducerThread,self).__init__()
        self.host = host

    def run(self):
        pbar = tqdm.tqdm(1000, position=1)
        while True:
            if not self.host.vid.isopened:
                break
            emit = self.host.emit()
            pbar.update(1)
            # try:
            # q.put(emit) #give up frames 
            q.put([None, None])
            # except Exception:
            #     pass
            
            # print('Producer q', q.empty())



class PostProcessThread(multiprocessing.Process):
    def __init__(self, queue):
        super(PostProcessThread,self).__init__()
        self.queue = queue
        
    def run(self):
        cv2.namedWindow("preview")
        pbar = tqdm.tqdm(1000, position=1, desc='CV show')
        while True:
            pbar.update(1)
            item = self.queue.get()
            show_kpt_data(*item)

def box2cs(box, image_size):
    """Encode bbox(x,y,w,h) into (center, scale) without padding.

    Returns:
        tuple: A tuple containing center and scale.
    """
    x, y, w, h = box[:4]

    aspect_ratio = 1. * image_size[0] / image_size[1]
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / 200.0, h * 1.0 / 200.0], dtype=np.float32)
    return center, scale
    

def get_max_preds(heatmaps:np.ndarray):
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def get_max_preds_gpu(heatmaps:torch.Tensor):
    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = torch.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = torch.amax(heatmaps_reshaped, 2).reshape((N, K, 1))
    idx = idx.cpu().numpy()
    maxvals = maxvals.cpu().numpy()

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def transform_preds(coords, center, scale, output_size, use_udp=False):
    assert coords.shape[-1] == 2
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    scale = scale * 200.0
    output_size = np.array(output_size)
    if use_udp:
        preds = scale / (output_size - 1.0)
    else:
        scale_step = scale / output_size
    preds = coords * scale_step + center - scale * 0.5
    return preds


class DataSet():
    def __init__(self, vid, cfg, views_xywh):
        self.vid = vid
        self.views_xywh = views_xywh
        _, _, w, h = views_xywh[0]

        bbox = np.array([0, 0, w, h])
        center, scale = box2cs(bbox, cfg.data_cfg['image_size'])
        self.center, self.scale = center, scale

        self.resize_pipelines = []
        def indentical(x):
            return x
        def crop(img, x, y, w, h):
            return img[y:y+h, x:x+w]

        for crop_xywh in views_xywh:
            x,y,w,h = crop_xywh
            fun_crop = partial(crop, x=x, y=y, w=w, h=h)
            old_size = [h, w]
            desired_size = cfg.data_cfg['image_size'][::-1] #(h,w)
            ratio = np.min( np.array(desired_size)/np.array(old_size) )
            expand_size = tuple([int(x*ratio) for x in old_size]) #(h,w)
            if expand_size != old_size:
                cv2_resize = partial(cv2.resize, dsize=(expand_size[1], expand_size[0]),
                        interpolation=cv2.INTER_LINEAR)
            else:
                cv2_resize = indentical
            
            top = (desired_size[0] - expand_size[0])//2
            bottom = desired_size[0] - expand_size[0] - top
            left= (desired_size[1] - expand_size[1])//2
            right = desired_size[1] - expand_size[1] - left
            if expand_size != desired_size:
                cv2_copyMakeBorder = partial(cv2.copyMakeBorder, top=top, bottom=bottom,
                     left=left, right=right, borderType=cv2.BORDER_CONSTANT, value=0)
            else:
                cv2_copyMakeBorder = indentical
            
            self.resize_pipelines.append([fun_crop, cv2_resize, cv2_copyMakeBorder])
            
        self.desired_size = desired_size  #(h,w)
        # producer = ProducerThread(self)
        # producer.start()

    def emit(self):
        img_NHWC = np.zeros((len(self.resize_pipelines), *self.desired_size, 3), dtype=np.uint8)
        ret, img = self.vid.read()
        if not ret: raise StopIteration
        img = img.copy()
        img_preview = cv2.resize(img, preview_resize, interpolation=cv2.INTER_LINEAR)
        for i, pannel_pipelines in enumerate(self.resize_pipelines):
            img_pannel = img
            for p in pannel_pipelines:
                img_pannel = p(img_pannel)
            img_HWC = img_pannel
            img_NHWC[i] = img_HWC
        img_NCHW = torch.from_numpy(img_NHWC).permute(0,3,1,2)
        return img_NCHW, img_preview

    def __iter__(self):
        while True:
            img_NCHW, img_preview = self.emit()
            yield img_NCHW, img_preview


class MyWorker:
    def __init__(self, config, checkpoint, ballcalib):
        super().__init__()
        self.id = getattr(self, 'id', 0)
        self.cuda = getattr(self, 'cuda', 0)
        pose_model = init_pose_model(
                        config, checkpoint=None, device='cpu')
        self.pose_model = pose_model
        self.checkpoint = checkpoint
        self.out_queue = multiprocessing.Queue(maxsize=2)
        self.postprocess = PostProcessThread(self.out_queue)
        # self.postprocess.start()
        self.calibobj = CalibPredict(ballcalib) if ballcalib else None
        print("well setup worker:", self.cuda)

    def compute(self, args):
        with torch.cuda.device(self.cuda):
            trt_model = TRTModule()
            trt_model.load_state_dict(torch.load(self.checkpoint))
            outdata = trt_model(torch.rand(4,3,512,512).cuda().half())
            
        model = self.pose_model
        cfg = model.cfg
 
        vid = ffmpegcv.VideoCaptureCAM("OBS Virtual Camera", 
            camsize=camsize, 
            pix_fmt='rgb24')

        device = torch.device('cuda:{}'.format(self.cuda))
        dataset = DataSet(vid, cfg, pos_views)
        center, scale = dataset.center, dataset.scale
        
        pbar = tqdm.tqdm(10000, desc='loading')
        # cv2.namedWindow("preview")
        for img_NCHW, img_preview  in dataset:
            pbar.update(1)
            kpt_data = []  #N,K,xyp
            batch_img = img_NCHW.to(device).half() #380fps
            with torch.cuda.device(self.cuda):
                heatmap = trt_model(batch_img)

            N, K, H, W = heatmap.shape
            preds, maxvals = get_max_preds_gpu(heatmap)
            preds = transform_preds(
                preds, center, scale, [W, H], use_udp=False)
            
            all_preds = np.concatenate((preds, maxvals), axis=-1)
            kpt_data.append(all_preds)
            
            kpt_data = np.array(kpt_data).squeeze()  #N, xyp
            show_kpt_data(camsize, kpt_data, pos_views, img_preview, self.calibobj)
            # self.out_queue.put([camsize, kpt_data, pos_views, img_preview, self.calibobj])
            

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
    parser.add_argument('--pannels', type=int, default=4, help='crop views')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--ballcalib', type=str, default=None)
    arg = parser.parse_args()

    pos_views[:] = get_view_xywh_wrapper(arg.pannels)
    config, checkpoint, ballcalib = arg.config, arg.checkpoint, arg.ballcalib
    if config is None:
        config = config_dict[arg.pannels]
    if checkpoint is None:
        checkpoint = findcheckpoint_trt(config, trtnake='latest.fullB4_fp16.trt')
    print("config:", config)
    print("checkpoint:", checkpoint)

    worker = MyWorker(config, checkpoint, ballcalib)
    worker.compute(None)

    