# python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_realtimecam --pannels 4 --config "E:\mmpose\res50_coco_ball_512x320_ZJF.py" --ballcalib "E:\mmpose\ball2.calibpkl"
# python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_realtimecam --pannels 9 --config "E:\mmpose\res50_coco_ball_512x320.py"
# %%
import argparse
import numpy as np
import tqdm
import torch
from mmpose.apis import init_pose_model
import ffmpegcv
from torch2trt import TRTModule
import cv2
from lilab.mmpose_dev.a2_convert_mmpose2trt import findcheckpoint_trt
from lilab.cameras_setup import get_view_xywh_wrapper
import itertools
from lilab.multiview_scripts_dev.s6_calibpkl_predict import CalibPredict


config_dict = {6:'/home/liying_lab/chenxinfeng/DATA/mmpose/hrnet_w32_coco_ball_512x512.py',
          10:'/home/liying_lab/chenxinfeng/DATA/mmpose/res50_coco_ball_512x512.py',
          4: r"E:\mmpose\res50_coco_ball_512x320_ZJF.py"}

pos_views = []
preview_resize = (1280, 800)
# camsize = [2560, 1600]
camsize = (3840, 2400)


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

        canvas_w, canvas_h = vid.width, vid.height
        desired_size = cfg.data_cfg['image_size'][::-1] #(h,w)
        chw_coord_ravel_nview = []
        c_channel_in=1
        c_channel_out=3
        assert c_channel_in==1 or c_channel_in==3
        for crop_xywh in views_xywh:
            chw_coord_ravel_nview.append(self.cv2_resize_idx_ravel((canvas_h, canvas_w), crop_xywh, desired_size, c_channel_in, c_channel_out))
        
        coord_NCHW_idx_ravel = np.array(chw_coord_ravel_nview)
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
        while True:
            ret, img = self.vid.read_gray()
            if not ret: raise StopIteration
            img_preview = cv2.resize(img, preview_resize, interpolation=cv2.INTER_NEAREST)
            if img_preview.ndim==2:
                img_preview = cv2.cvtColor(img_preview, cv2.COLOR_GRAY2BGR)
            elif img_preview.shape[-1]==1:
                img_preview = np.ascontiguousarray(np.repeat(img_preview, 3, axis=-1))
            # img_preview = np.zeros((preview_resize[1],preview_resize[0],3), dtype=np.uint8)
            # img_preview = img.ravel()[self.coord_preview_HWC_idx_ravel]
            img_NCHW = img.ravel()[self.coord_NCHW_idx_ravel]
            # img_N1HW = img.ravel()[self.coord_N1HW_idx_ravel]
            # img_NCHW = np.broadcast_to(img_N1HW, self.coord_NCHW_idx_ravel.shape)
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
        self.calibobj = CalibPredict(ballcalib) if ballcalib else None

    def compute(self, args):
        cfg = self.pose_model.cfg
        vid = ffmpegcv.VideoCaptureCAM("OBS Virtual Camera", 
            camsize=camsize, pix_fmt='nv12')
        # vid = ffmpegcv.VideoCaptureNV(r"F:\ball2.mp4", pix_fmt='nv12')
        # vid = ffmpegcv.VideoCaptureNV(r"E:\ball_9pannel.mp4", pix_fmt='nv12')
        print(vid.width, vid.height)
        assert (vid.width, vid.height) == camsize
        dataset = DataSet(vid, cfg, pos_views)
        dataset_iter = iter(dataset)
        center, scale = dataset.center, dataset.scale
        print("Well setup VideoCapture")

        count_range = range(dataset.__len__()) if hasattr(dataset, '__len__') else itertools.count()
        pbar = tqdm.tqdm(10000, desc='loading')

        with torch.cuda.device(self.cuda):
            from torch2trt.torch2trt import torch_dtype_from_trt
            trt_model = TRTModule()
            trt_model.load_from_engine(self.checkpoint)
            idx = trt_model.engine.get_binding_index(trt_model.input_names[0])
            input_dtype = torch_dtype_from_trt(trt_model.engine.get_binding_dtype(idx))
            input_shape = tuple(trt_model.context.get_binding_shape(idx))
            assert input_shape==dataset.coord_NCHW_idx_ravel.shape
            img_NCHW = np.zeros(input_shape)
            img_preview = np.zeros((*preview_resize,3))
            heatmap = mid_gpu(trt_model, img_NCHW, input_dtype)

            for idx, _ in enumerate(count_range, start=-1):
                pbar.update(1)
                heatmap_wait = mid_gpu(trt_model, img_NCHW, input_dtype)
                img_NCHW_next, img_preview_next = pre_cpu(dataset_iter)
                torch.cuda.current_stream().synchronize()
                heatmap = heatmap_wait
                if idx<=-1: continue
                post_cpu(heatmap, center, scale, pos_views, img_preview, self.calibobj)
                img_NCHW, img_preview = img_NCHW_next, img_preview_next

            heatmap = mid_gpu(trt_model, img_NCHW)
            post_cpu(heatmap, center, scale, pos_views, img_preview, self.calibobj)


def pre_cpu(dataset_iter):
    img_NCHW, img_preview = next(dataset_iter)
    return img_NCHW, img_preview


def mid_gpu(trt_model, img_NCHW, input_dtype):
    batch_img = torch.from_numpy(img_NCHW).cuda().type(input_dtype)
    heatmap = trt_model(batch_img)
    return heatmap


def post_cpu(heatmap, center, scale, pos_views, img_preview, calibobj):
    N, K, H, W = heatmap.shape
    preds, maxvals = get_max_preds_gpu(heatmap)
    preds = transform_preds(
                preds, center, scale, [W, H], use_udp=False)
    kpt_data = np.concatenate((preds, maxvals), axis=-1) #(N, xyp)
    show_kpt_data(camsize, kpt_data, pos_views, img_preview, calibobj)


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
        checkpoint = findcheckpoint_trt(config, trtnake='latest.full_fp16.engine')
    print("config:", config)
    print("checkpoint:", checkpoint)

    worker = MyWorker(config, checkpoint, ballcalib)
    worker.compute(None)
