import numpy as np
import pickle
from multiview_calib.calibpkl_predict import CalibPredict
from lilab.dannce_realtime.dannce_realtime import get_voxel_size, DataGenerator_3Dconv_torch
from typing import List, Dict, Text, AnyStr
import cv2
from dannce.engine import ops
from dannce.engine.inference_cxf import mid_gpu
from dannce.my_extension_voxel import voxel_sampling_c_last
from dannce.engine import processing_cxf as processing
import ffmpegcv
import torch
import tqdm
import itertools
from lilab.multiview_scripts.rat2d_kptvideo import cv_plot_skeleton_aframe
import os
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing import Queue
from multiprocessing.synchronize import Lock
from lilab.yolo_seg.common_variable import (
    NFRAME, out_numpy_imgNNHW_shape, out_numpy_com2d_shape, out_numpy_previ_shape)
from lilab.yolo_seg.sockerServer import p3d, p2d

ballfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/carl/2023-10-14-/ball_2023-10-23_13-18-10.calibpkl'
nclass = 2


class DataGenerator_3Dconv_torch_video_canvas_multivoxel(DataGenerator_3Dconv_torch):
    def set_video(self, coms_3d:np.ndarray, calibobj:CalibPredict,
                  shared_array_imgNNHW:SynchronizedArray,
                  shared_array_com2d:SynchronizedArray,
                  shared_array_previ:SynchronizedArray,
                  q:Queue, lock:Lock):
        
        self.numpy_imgNNHW = np.frombuffer(shared_array_imgNNHW.get_obj(), dtype=np.uint8).reshape((NFRAME,*out_numpy_imgNNHW_shape))
        self.numpy_com2d = np.frombuffer(shared_array_com2d.get_obj(), dtype=np.float64).reshape((NFRAME, *out_numpy_com2d_shape))
        self.numpy_previ = np.frombuffer(shared_array_previ.get_obj(), dtype=np.uint8).reshape((NFRAME, *out_numpy_previ_shape))
        self.calibobj = calibobj
        self.q = q
        self.lock = lock

        assert self.batch_size == 1, "Batch size must be 1 for video data"

        # assert tuple(self.canvas_hw) == (ch, cw)
        self.ncam = out_numpy_imgNNHW_shape[0]
        self.image_hw = out_numpy_imgNNHW_shape[-2:]
        self.batch_size = nclass
        self.n_channels_in = 1 #gray

        self.voxel_size_list = self.kwargs.get("vol_size_list", None)
        if self.voxel_size_list is None or self.voxel_size_list is []:
            self.voxel_size_list = [self.kwargs["vol_size"]]*nclass
        else:
            assert len(self.voxel_size_list) == nclass

        assert len(set(self.voxel_size_list))==1
        self.init_grids_one(coms_3d)

    def decode_array(self):
        data_id = self.q.get()
        with self.lock:
            img_NNHW = self.numpy_imgNNHW[data_id]
            com2d = self.numpy_com2d[data_id]
            img_HW = self.numpy_previ[data_id]
        com3d = self.calibobj.p2d_to_p3d(com2d)
        img_CBHW = img_NNHW.transpose(1,0,2,3)
        return img_CBHW, img_HW, com3d


    def __init__(self, *args, **kwargs):
        self.iframe = 0
        self.ipannel_preview = 0
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        indexes = np.arange(self.batch_size) + index * self.batch_size
        list_IDs_temp = [f'0_{i}' for i in indexes]
        self.batch_size = nclass
        img_NBHW, img_HW, coms_3d_nclass = self.decode_array()
        coms_3d_nclass[np.isnan(coms_3d_nclass)] = 0
        coms_3d_nclass = np.clip(self.com3d_minmax[0], self.com3d_minmax[1], coms_3d_nclass)
        im_pannels_nclass = img_NBHW[...,None] #NBHWC
        assert len(im_pannels_nclass) == self.batch_size == len(list_IDs_temp)
        assert not self.depth
        assert self.mode == "3dprob"
        assert self.mono

        xgrid_roi, y_3d = [], []
        X = np.zeros((nclass, self.nvox, self.nvox, self.nvox, self.ncam))
        xgrid_roi = np.zeros((nclass, self.nvox, self.nvox, self.nvox, 3), dtype=np.float16)
        for i in range(nclass):
            ims = im_pannels_nclass[i]
            com_3d = coms_3d_nclass[i] #to be determined
            [X[i], xgrid_roi[i]], y_3d_each = self.quary_gridsample_by_com3d(com_3d, ims, i) #cxf
            y_3d.append(y_3d_each)

        self.iframe += 1
        iview_preview = cv2.cvtColor(img_HW, cv2.COLOR_GRAY2BGR) 
        return [X, xgrid_roi], y_3d, iview_preview

    def init_grids_one(self, com_3ds):
        self.grid_1d_l = []
        self.grid_coord_3d_l = []
        self.proj_grid_voxel_ncam_indravel_l = []

        voxel_size = self.voxel_size_list[0]
        self.vmin, self.vmax = -voxel_size / 2, voxel_size / 2
        vstep = (self.vmax - self.vmin) / self.nvox
        assert com_3ds.shape[1]==3
        Xaxis_min, Yaxis_min, Zaxis_min = com_3ds.min(axis=0)
        Xaxis_max, Yaxis_max, Zaxis_max = com_3ds.max(axis=0)
        self.com3d_minmax = np.array([[Xaxis_min, Yaxis_min, Zaxis_min],
                                      [Xaxis_max, Yaxis_max, Zaxis_max]])
        xgrid = np.arange(Xaxis_min+1.2*self.vmin, Xaxis_max+1.2*self.vmax, vstep)
        ygrid = np.arange(Yaxis_min+1.2*self.vmin, Yaxis_max+1.2*self.vmax, vstep)
        zgrid = np.arange(Zaxis_min+1.2*self.vmin, Zaxis_max+1.2*self.vmax, vstep)
        (x_coord_3d, y_coord_3d, z_coord_3d) = np.meshgrid(xgrid, ygrid, zgrid)
        grid_flatten_3d = np.stack((x_coord_3d.ravel(), y_coord_3d.ravel(), z_coord_3d.ravel()),axis=1)
        experimentID=0

        camParams = [self.camera_params[experimentID][name] for name in self.camnames[experimentID]]
        assert len(camParams) == self.ncam
        proj_grid_voxel_ncam = []
        for camParam in camParams:
            proj_grid = ops.project_to2d(grid_flatten_3d, camParam["K"], camParam["R"], camParam["t"])
            proj_grid = proj_grid[:, :2]
            if self.distort:
                proj_grid = ops.distortPoints(
                    proj_grid,
                    camParam["K"],
                    np.squeeze(camParam["RDistort"]),
                    np.squeeze(camParam["TDistort"]),
                ).T
            proj_grid_voxel = np.reshape(proj_grid, [*x_coord_3d.shape, 2]).astype('float16')
            proj_grid_voxel_ncam.append(proj_grid_voxel)
            
        self.proj_grid_voxel_ncam = np.array(proj_grid_voxel_ncam).astype('int16')  #(ncam, nvox_y, nvox_x, nvox_z, 2=hw)
        self.proj_grid_voxel_ncam_indravel = np.zeros([*self.proj_grid_voxel_ncam.shape[:-1]], dtype='int64') #(ncam, nvox_y, nvox_x, nvox_z)
        for i in range(self.ncam):
            np.clip(self.proj_grid_voxel_ncam[i, ..., 0], 0, self.image_hw[1] - 1, out=self.proj_grid_voxel_ncam[i, ..., 0])
            np.clip(self.proj_grid_voxel_ncam[i, ..., 1], 0, self.image_hw[0] - 1, out=self.proj_grid_voxel_ncam[i, ..., 1])
            indravel = np.ravel_multi_index(np.moveaxis(self.proj_grid_voxel_ncam[i, ..., ::-1], -1, 0), self.image_hw)
            self.proj_grid_voxel_ncam_indravel[i] = indravel[...]

        self.grid_1d = (xgrid, ygrid, zgrid)
        self.grid_coord_3d = np.stack([x_coord_3d, y_coord_3d, z_coord_3d], axis=-1).astype('float16')  #(nvox_y, nvox_x, nvox_z, 3)
            
        for voxel_size in self.voxel_size_list:
            self.grid_1d_l.append(self.grid_1d)
            self.grid_coord_3d_l.append(self.grid_coord_3d)
            self.proj_grid_voxel_ncam_indravel_l.append(self.proj_grid_voxel_ncam_indravel)

    def quary_gridsample_by_com3d(self, com_3d:np.ndarray, ims:np.ndarray, iclass:int):
        # input=gray, output=gray.
        assert len(com_3d)==3
        assert len(ims) == self.ncam
        grid_coord_3d = self.grid_coord_3d_l[iclass]
        grid_1d = self.grid_1d_l[iclass]
        proj_grid_voxel_ncam_indravel = self.proj_grid_voxel_ncam_indravel_l[iclass]
        com_index = np.array([np.searchsorted(grid_1d[i], com_3d[i], side='right')
                        for i in range(3)])  #(3,)
        com_range = np.floor(com_index[:,None] + [- self.nvox/2, self.nvox/2]).astype(int) #(3,2)

        xgrid_roi = grid_coord_3d[ com_range[1][0]:com_range[1][1], 
                                   com_range[0][0]:com_range[0][1], 
                                   com_range[2][0]:com_range[2][1],
                                   :]   #(nvox_y, nvox_x, nvox_z, 3)

        result = voxel_sampling_c_last(ims[...,0], proj_grid_voxel_ncam_indravel, com_range[[1,0,2]][:,0], (self.nvox, self.nvox, self.nvox)) #nvox, nvox, nvox, nview
        X = result.astype(np.float32) #(nvox, nvox, nvox, nview)
        assert self.norm_im
        X = processing.preprocess_3d(X)
        y_3d = np.empty((self.nvox, self.nvox, self.nvox, 14), dtype="float32")

        return [X, xgrid_roi], y_3d


def infer_dannce_max_trt(
    generator: DataGenerator_3Dconv_torch_video_canvas_multivoxel,
    model,
    device: Text,
    calibPredict: CalibPredict
):
    from torch2trt.torch2trt import torch_dtype_from_trt
    
    vidout = ffmpegcv.VideoWriter('/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/out.mkv', codec='h264', fps=30)
    # vidout = ffmpegcv.VideoWriterStreamRT('rtmp://localhost:1935/mystream2')
    with torch.cuda.device(device):
        # model warmup
        idx = model.engine.get_binding_index(model.input_names[0])
        dtype = torch_dtype_from_trt(model.engine.get_binding_dtype(idx))
        shape = tuple(model.context.get_binding_shape(idx))
        if shape[0]==-1: shape = (1, *shape[1:])
        input = torch.empty(size=shape, dtype=dtype).cuda()
        output = model(input)
        dtype = output.dtype
        X, X_grid = input.cpu().numpy(), np.zeros((*shape[:-1], 2), dtype='float16')
        pannel_preview = None
        for iframe in tqdm.tqdm(itertools.count(-1), position=2, desc='VoxelPrediction'):
            pred_wait = mid_gpu(X, dtype, model)
            X_next, X_grid_next, pannel_preview_next = pre_cpu(generator, iframe)
            torch.cuda.current_stream().synchronize()
            pred = pred_wait
            post_cpu(pred, X_grid, iframe, pannel_preview, calibPredict, vidout)
            X, X_grid, pannel_preview = X_next, X_grid_next, pannel_preview_next


def pre_cpu(generator, i):
    [X, X_grid], y, pannel_preview  = generator[i]
    return X, X_grid, pannel_preview


def post_cpu(pred, X_grid, iframe, pannel_preview, calibPredict:CalibPredict, vidout):
    if iframe<0: return
    nclass,n1,n2,n3, k = pred.shape
    ind_max = torch.reshape(pred, (nclass, n1*n2*n3, k)).argmax(axis=1).cpu().numpy()
    # ind_max = pred.cpu().numpy()
    com_3d_l = X_grid[:,[0,-1], [0,-1], [0,-1]].mean(axis=1)    #(nclass, 3)
    p3d[:] = np.take_along_axis(X_grid.reshape(nclass, n1*n2*n3, 1, 3), 
                             ind_max[:,None,None:,None], axis=1)[:,0,:] #(nclass, k, 3)
    ipannel = 1
    com_2d_l = calibPredict.p3d_to_p2d(com_3d_l)[ipannel].astype(np.int32)
    p2d[:] = calibPredict.p3d_to_p2d(p3d).astype(int)
    coord_2d = p2d[ipannel]
    pts2d_b_now, pts2d_w_now = coord_2d[0], coord_2d[1]
    frame = pannel_preview
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] #b,g,r
    for ianimal in range(len(com_2d_l)):
        cv2.circle(frame, tuple(com_2d_l[ianimal]), 10, colors[ianimal], -1)

    frame = cv_plot_skeleton_aframe(frame, pts2d_b_now, name = 'black')
    frame = cv_plot_skeleton_aframe(frame, pts2d_w_now, name = 'white')
    vidout.write(frame)


def dannce_predict_video_trt(params:Dict, ba_poses:Dict, model_file:AnyStr,
                             shared_array_imgNNHW:SynchronizedArray,
                             shared_array_com2d:SynchronizedArray,
                             shared_array_previ:SynchronizedArray,
                             q:Queue, lock:Lock):
    """Predict with dannce network

    Args:
        params (Dict): Paremeters dictionary.
    """
    from dannce.utils_cxf.cameraIntrinsics_OpenCV import cv2_pose_to_matlab_pose
    from collections import OrderedDict
    from torch2trt import TRTModule
    # Save 
    
    params['crop_width'] = np.array(params['crop_width']).tolist()
    params['crop_height'] = np.array(params['crop_height']).tolist()

    # Depth disabled until next release.
    params["depth"] = False
    n_views = int(params["n_views"])
    gpu_id = 1
    device = f'cuda:{gpu_id}'
    params["base_exp_folder"] = '.'
    datadict = {}      #label:= filenames
    datadict_3d = {}   #3d xyz
    com3d_dict = {}
    cameras = {}
    camnames = {}
    params["experiment"] = {}

    voxellength, coms_3d_fake = get_voxel_size(model_file)
    params["vol_size"] = voxellength
    print(f"Use real voxellength: {voxellength} mm")
    params["vmin"] = params["vmax"] = voxellength//2

    camParams = cv2_pose_to_matlab_pose(ba_poses)
    calibPredict = CalibPredict({'ba_poses': ba_poses})
    ncamera = len(camParams)
    assert ncamera == n_views, 'ncamera != n_views'
    e=0
    camnames[e] = [f'Camera{i+1}' for i in range(ncamera)]
    cameras[e] = OrderedDict(zip(camnames[e], camParams))

    vids = None
    params['camParams'] = camParams

    # For real mono prediction
    params["chan_num"] = 1

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
    if "vol_size_list" in params and params["vol_size_list"]:
        valid_params["vol_size_list"] = params["vol_size_list"]
    else:
        valid_params['vol_size'] = params['vol_size']

    # Datasets
    partition = {"valid_sampleIDs": [f'0_{i}' for i in range(180000)]}

    # TODO: Remove tifdirs arguments, which are deprecated
    tifdirs = []

    # Generate the dataset
    valid_generator = DataGenerator_3Dconv_torch_video_canvas_multivoxel(
        partition["valid_sampleIDs"],
        datadict,
        datadict_3d,
        cameras,
        partition["valid_sampleIDs"],
        com3d_dict,
        tifdirs,
        **valid_params
    )
    valid_generator.set_video(coms_3d_fake, calibPredict,
                              shared_array_imgNNHW, shared_array_com2d,
                              shared_array_previ,
                              q, lock)

    # Load model from tensorrt
    assert params["dannce_predict_model"] is not None
    assert params["predict_mode"] == "torch"
    assert not params["expval"]

    mdl_file = params["dannce_predict_model"]
    mdl_file = mdl_file.replace('.hdf5', '.engine')
    print("Loading model from " + mdl_file)
    assert os.path.exists(mdl_file), f"Model file {mdl_file} not found"

    
    with torch.cuda.device(device):
        trt_model = TRTModule()
        trt_model.load_from_engine(mdl_file)

    assert not params["expval"]

    infer_dannce_max_trt(
        valid_generator,
        trt_model,
        device,
        calibPredict
    )
