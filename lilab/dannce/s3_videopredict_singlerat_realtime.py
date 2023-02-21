# python -m lilab.dannce.s2_videopredict_singlerat_com3d --pannel 4 --video_file xx.mp4 --ballcalib xxx.calib --config xx.cfg 
# similar to lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_realtimecam
"""
cd /home/liying_lab/chenxinfeng/DATA/mmpose
python -m lilab.dannce.s2_videopredict_singlerat_realtime --pannel 4 \
    --video_file /mnt/liying.cibr.ac.cn_Data_Temp/ZJF_lab/0921/male.mp4 \
    --ballcalib /mnt/liying.cibr.ac.cn_Data_Temp/ZJF_lab/ball2.calibpkl \
    --com3d_config res50_coco_com2d_512x320_ZJF.py
"""
import argparse
import numpy as np
import tqdm
import yaml
import torch
from torch2trt import TRTModule
import os
import pickle
from collections import OrderedDict
from lilab.mmpose_dev.a2_convert_mmpose2engine import findcheckpoint_trt
from lilab.cameras_setup import get_view_xywh_wrapper
import itertools
from sklearn.impute import KNNImputer
from torch2trt.torch2trt import torch_dtype_from_trt
from scipy.ndimage import convolve1d
from lilab.dannce.cameraIntrinsics_OpenCV import cv2_pose_to_matlab_pose
from dannce.engine.generator_cxf import DataGenerator_3Dconv_torch_video_canvas_faster_single as DataGenerator_3Dconv_torch_video_single

from lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_realtimecam import (
    get_max_preds_gpu, transform_preds, pre_cpu, mid_gpu,
    preview_resize, MyWorker as OldMyWorker
)

com3d_config_dict = {6:'/home/liying_lab/chenxinfeng/DATA/mmpose/hrnet_w32_coco_ball_512x512.py',
          10:'/home/liying_lab/chenxinfeng/DATA/mmpose/res50_coco_ball_512x512.py',
          4: r"E:\mmpose\res50_coco_ball_512x320_ZJF.py"}

voxel_config_dict = {4: r"/home/liying_lab/chenxinfeng/DATA/dannce/demo_single/rat14_1280x800x4_mono/params.yaml"}

class DataLoader_video_canvas_faster_single(DataGenerator_3Dconv_torch_video_single):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def set_video(self, noneuse, gpu, pkldata):
        self.nclass = 1
        self.batch_size = self.nclass
        self.n_channels_in = 1 #gray
        self.gpu = gpu
        self.device = torch.device(f"cuda:{gpu}")
    
        coms_3d = np.array([[-20, -20, -10],
                    [20, 20, 10]])
        coms_3d = coms_3d.reshape(-1, coms_3d.shape[-1])
        self.init_grids(coms_3d)

    def __getitem__(self, index):
        raise NotImplementedError

    def quary_voxel(self, ims_pannel, com_3d):
        assert com_3d.shape == (3,)
        im_pannels_nclass = np.array(ims_pannel)
        assert len(im_pannels_nclass) == self.batch_size == 1
        assert not self.depth
        assert self.mode == "3dprob"
        assert self.mono

        X, xgrid_roi, y_3d = [], [], []
        for i, ID in enumerate(range(self.nclass)):
            ims = im_pannels_nclass[i]
            [X_each, xgrid_roi_each], y_3d_each = self.quary_gridsample_by_com3d(com_3d, ims)
            X.append(X_each)
            xgrid_roi.append(xgrid_roi_each)
            y_3d.append(y_3d_each)
        X = np.stack(X, axis=0)
        return [X, xgrid_roi], y_3d


def read_voxel_params(voxel_config, ballcalib):
    params = yaml.load(open(voxel_config), Loader=yaml.FullLoader)
    ballpkldata = pickle.load(open(ballcalib, 'rb'))
    camParams = cv2_pose_to_matlab_pose(ballpkldata['ba_poses'])

    params["depth"] = False
    n_views = int(params["n_views"])
    gpu_id = 0
    device = f'cuda:{gpu_id}'
    params["base_exp_folder"] = '.'
    datadict = {}      #label:= filenames
    datadict_3d = {}   #3d xyz
    com3d_dict = {}
    cameras = {}
    camnames = {}
    params["experiment"] = {}
    ncamera = len(camParams)
    assert ncamera == n_views, 'ncamera != n_views'
    e=0
    camnames[e] = [f'Camera{i+1}' for i in range(ncamera)]
    cameras[e] = OrderedDict(zip(camnames[e], camParams))
    vids = None
    params['camParams'] = camParams

    # For real mono prediction
    params["chan_num"] = 1 if params["mono"] else params["n_channels_in"]

    # Parameters
    valid_params = {
        "dim_in": (
            params["crop_height"][1] - params["crop_height"][0],
            params["crop_width"][1] - params["crop_width"][0],
        ),
        "n_channels_in": params["n_channels_in"],
        "batch_size": 1,
        "n_channels_out": params["n_channels_out"],
        "out_scale": params["sigma"],
        "crop_width": params["crop_width"],
        "crop_height": params["crop_height"],
        "vmin": params["vmin"],
        "vmax": params["vmax"],
        "nvox": params["nvox"],
        "interp": params["interp"],
        "mode": "coordinates" if params["expval"] else "3dprob",
        "camnames": camnames,
        "shuffle": False,
        "rotation": False,
        "vidreaders": vids,
        "distort": True,
        "expval": params["expval"],
        "crop_im": False,
        "mono": params["mono"],
        "mirror": False,
        "predict_flag": True,
        "gpu_id": str(gpu_id),
    }

    partition = {"valid_sampleIDs": [f'0_{i}' for i in range(180000)]}

    tifdirs = []
    valid_generator = genfunc(
        partition["valid_sampleIDs"],
        datadict,
        datadict_3d,
        cameras,
        partition["valid_sampleIDs"],
        com3d_dict,
        tifdirs,
        **valid_params
    )
    genfunc = DataGenerator_3Dconv_torch_video_single
    valid_generator.set_video(None, gpu_id, ballpkldata)
    return valid_generator


class MyWorker(OldMyWorker):
    def __init__(self, com3d_config, video_file, com3d_checkpoint, ballcalib, views_xywh, voxel_config, voxel_checkpoint):
        super().__init__(com3d_config, video_file, com3d_checkpoint, ballcalib, views_xywh)
        assert os.path.exists(voxel_checkpoint)
        assert os.path.exists(voxel_config)


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
            calibobj = self.calibobj
            keypoints_xyz_ba, keypoints_xyp = [], []
            for idx, _ in enumerate(count_range, start=-1):
                pbar.update(1)
                heatmap_wait = mid_gpu(trt_model, img_NCHW, input_dtype)
                img_NCHW_next, img_preview_next = pre_cpu(dataset_iter)
                torch.cuda.current_stream().synchronize()
                heatmap = heatmap_wait
                if idx<=-1: continue
                # if idx>5000: break
                kpt3d, kpt2d = post_cpu(self.camsize, heatmap, center, scale, views_xywh, img_preview, calibobj)
                keypoints_xyz_ba.append(kpt3d)
                keypoints_xyp.append(kpt2d)
                img_NCHW, img_preview = img_NCHW_next, img_preview_next



def post_cpu(camsize, heatmap, center, scale, views_xywh, img_preview, calibobj):
    N, K, H, W = heatmap.shape
    preds, maxvals = get_max_preds_gpu(heatmap)
    preds = transform_preds(
                preds, center, scale, [W, H], use_udp=False)
    keypoints_xyp = np.concatenate((preds, maxvals), axis=-1) #(N, xyp)

    # thr
    thr = 0.4    
    indmiss = keypoints_xyp[...,2] < thr
    keypoints_xyp[indmiss] = np.nan
    keypoints_xy = keypoints_xyp[...,:2]

    # ba
    if calibobj is not None:
        keypoints_xyz_ba = calibobj.p2d_to_p3d(keypoints_xy)
        # keypoints_xy_ba = calibobj.p3d_to_p2d(keypoints_xyz_ba)
    else:
        keypoints_xyz_ba = np.ones((*keypoints_xy.shape[1:-1], 3)) * np.nan
        # keypoints_xy_ba = keypoints_xy * np.nan
    return keypoints_xyz_ba, keypoints_xyp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pannels', type=int, default=4, help='crop views')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--com3d_config', type=str, default=None)
    parser.add_argument('--com3d_checkpoint', type=str, default=None)
    parser.add_argument('--voxel_config', type=str, default=None)
    parser.add_argument('--voxel_checkpoint', type=str, default=None)
    parser.add_argument('--ballcalib', type=str, default=None)
    arg = parser.parse_args()

    views_xywh = get_view_xywh_wrapper(arg.pannels)
    com3d_config, com3d_checkpoint, ballcalib = arg.com3d_config, arg.com3d_checkpoint, arg.ballcalib
    if com3d_config is None:
        com3d_config = com3d_config_dict[arg.pannels]
    if com3d_checkpoint is None:
        com3d_checkpoint = findcheckpoint_trt(com3d_config, trtnake='latest.full_fp16.engine')
    assert arg.video_file, "--video_file should be set."
    assert ballcalib, "--ballcalib should be set"
    print("com3d_config:", com3d_config)
    print("com3d_checkpoint:", com3d_checkpoint)
    
    voxel_config, voxel_checkpoint = arg.voxel_config, arg.voxel_checkpoint
    if voxel_config is None:
        voxel_config = voxel_config_dict[arg.pannels]
    if voxel_checkpoint is None:
        voxel_checkpoint = os.path.dirname(com3d_checkpoint) + '/DANNCE/train_results/MAX/latest_fp16.engine'

    worker = MyWorker(com3d_config, arg.video_file, com3d_checkpoint, ballcalib, views_xywh, voxel_config, voxel_checkpoint)
    worker.compute(None)
