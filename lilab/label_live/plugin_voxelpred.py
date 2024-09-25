import numpy as np
import pickle
from multiview_calib.calibpkl_predict import CalibPredict
from lilab.dannce_realtime.dannce_realtime import (
    get_voxel_size,
    DataGenerator_3Dconv_torch,
)
from typing import List, Dict, Text, AnyStr
import cv2
from dannce.engine import ops
from dannce.my_extension_voxel import voxel_sampling_c_last
from dannce.engine import processing_cxf as processing
import ffmpegcv
import torch
import tqdm
import itertools
import ctypes
from lilab.multiview_scripts.rat2d_kptvideo import cv_plot_skeleton_aframe
import os
import multiprocessing
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing import Queue
from multiprocessing.synchronize import Lock
from lilab.yolo_seg.common_variable import (
    NFRAME,
    get_numpy_handle,
    create_shared_arrays_dannce,
    get_numpy_handle_dannce,
)
from lilab.label_live.biLSTM_behavior_classify import cluster_main
from lilab.timecode_tag.netcoder import Netcoder
from ffmpegcv.ffmpeg_writer_noblock import FFmpegWriterNoblock
from lilab.label_live.water_timecode import water_timecode



nclass = 2


def get_vidout():
    vidout = ffmpegcv.VideoWriterStreamRT(
        "rtsp://10.50.60.6:8554/mystream_behaviorlabel_result"
    )
    return vidout


class DataGenerator_3Dconv_torch_video_canvas_multivoxel(DataGenerator_3Dconv_torch):
    def set_video(
        self,
        coms_3d: np.ndarray,
        calibobj: CalibPredict,
        numpy_imgNKHW: SynchronizedArray,
        numpy_com2d: SynchronizedArray,
        numpy_previ: SynchronizedArray,
        q: Queue,
        lock: Lock,
    ):

        self.numpy_imgNKHW = numpy_imgNKHW
        self.numpy_com2d = numpy_com2d
        self.numpy_previ = numpy_previ
        self.calibobj = calibobj
        self.q = q
        self.lock = lock

        assert self.batch_size == 1, "Batch size must be 1 for video data"

        # assert tuple(self.canvas_hw) == (ch, cw)
        self.ncam = numpy_imgNKHW.shape[1]
        self.image_hw = numpy_imgNKHW.shape[-2:]
        self.batch_size = nclass
        self.n_channels_in = 1  # gray

        self.voxel_size_list = self.kwargs.get("vol_size_list", None)
        if self.voxel_size_list is None or self.voxel_size_list is []:
            self.voxel_size_list = [self.kwargs["vol_size"]] * nclass
        else:
            assert len(self.voxel_size_list) == nclass

        assert len(set(self.voxel_size_list)) == 1
        self.init_grids_one(coms_3d)

    def decode_array(self):
        data_id = self.q.get()
        self.buffer_index = data_id
        if data_id is None:
            return None, None, None
        with self.lock:
            pass
        img_HW = self.numpy_previ[data_id]
        com3d = self.calibobj.p2d_to_p3d(self.numpy_com2d[data_id])
        img_CBHW = self.numpy_imgNKHW[data_id].transpose(1, 0, 2, 3)
        return img_CBHW, img_HW, com3d

    def __init__(self, *args, **kwargs):
        self.iframe = 0
        self.ipannel_preview = 0
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        self.batch_size = nclass
        img_NBHW, img_HW, coms_3d_nclass = self.decode_array()
        if img_NBHW is None:
            return [None, None], None, None
        coms_3d_nclass[np.isnan(coms_3d_nclass)] = 0
        coms_3d_nclass = np.clip(
            self.com3d_minmax[0], self.com3d_minmax[1], coms_3d_nclass
        )
        im_pannels_nclass = img_NBHW[..., None]  # NBHWC
        assert len(im_pannels_nclass) == self.batch_size
        assert not self.depth
        assert self.mode == "3dprob"
        assert self.mono

        xgrid_roi, y_3d = [], []
        X = np.zeros((nclass, self.nvox, self.nvox, self.nvox, self.ncam))
        xgrid_roi = np.zeros(
            (nclass, self.nvox, self.nvox, self.nvox, 3), dtype=np.float16
        )
        for i in range(nclass):
            ims = im_pannels_nclass[i]
            com_3d = coms_3d_nclass[i]  # to be determined
            [X[i], xgrid_roi[i]], y_3d_each = self.quary_gridsample_by_com3d(
                com_3d, ims, i
            )  # cxf
            y_3d.append(y_3d_each)

        self.iframe += 1
        iview_preview = img_HW
        return X, xgrid_roi, iview_preview

    def init_grids_one(self, com_3ds):
        self.grid_1d_l = []
        self.grid_coord_3d_l = []
        self.proj_grid_voxel_ncam_indravel_l = []

        voxel_size = self.voxel_size_list[0]
        self.vmin, self.vmax = -voxel_size / 2, voxel_size / 2
        vstep = (self.vmax - self.vmin) / self.nvox
        assert com_3ds.shape[1] == 3
        Xaxis_min, Yaxis_min, Zaxis_min = com_3ds.min(axis=0)
        Xaxis_max, Yaxis_max, Zaxis_max = com_3ds.max(axis=0)
        self.com3d_minmax = np.array(
            [[Xaxis_min, Yaxis_min, Zaxis_min], [Xaxis_max, Yaxis_max, Zaxis_max]]
        )
        xgrid = np.arange(
            Xaxis_min + 1.2 * self.vmin, Xaxis_max + 1.2 * self.vmax, vstep
        )
        ygrid = np.arange(
            Yaxis_min + 1.2 * self.vmin, Yaxis_max + 1.2 * self.vmax, vstep
        )
        zgrid = np.arange(
            Zaxis_min + 1.2 * self.vmin, Zaxis_max + 1.2 * self.vmax, vstep
        )
        (x_coord_3d, y_coord_3d, z_coord_3d) = np.meshgrid(xgrid, ygrid, zgrid)
        grid_flatten_3d = np.stack(
            (x_coord_3d.ravel(), y_coord_3d.ravel(), z_coord_3d.ravel()), axis=1
        )
        experimentID = 0

        camParams = [
            self.camera_params[experimentID][name]
            for name in self.camnames[experimentID]
        ]
        assert len(camParams) == self.ncam, f"camera {len(camParams)} != {self.ncam}"
        proj_grid_voxel_ncam = []
        for camParam in camParams:
            proj_grid = ops.project_to2d(
                grid_flatten_3d, camParam["K"], camParam["R"], camParam["t"]
            )
            proj_grid = proj_grid[:, :2]
            if self.distort:
                proj_grid = ops.distortPoints(
                    proj_grid,
                    camParam["K"],
                    np.squeeze(camParam["RDistort"]),
                    np.squeeze(camParam["TDistort"]),
                ).T
            proj_grid_voxel = np.reshape(proj_grid, [*x_coord_3d.shape, 2]).astype(
                "float16"
            )
            proj_grid_voxel_ncam.append(proj_grid_voxel)

        self.proj_grid_voxel_ncam = np.array(proj_grid_voxel_ncam).astype(
            "int16"
        )  # (ncam, nvox_y, nvox_x, nvox_z, 2=hw)
        self.proj_grid_voxel_ncam_indravel = np.zeros(
            [*self.proj_grid_voxel_ncam.shape[:-1]], dtype="int64"
        )  # (ncam, nvox_y, nvox_x, nvox_z)
        for i in range(self.ncam):
            np.clip(
                self.proj_grid_voxel_ncam[i, ..., 0],
                0,
                self.image_hw[1] - 1,
                out=self.proj_grid_voxel_ncam[i, ..., 0],
            )
            np.clip(
                self.proj_grid_voxel_ncam[i, ..., 1],
                0,
                self.image_hw[0] - 1,
                out=self.proj_grid_voxel_ncam[i, ..., 1],
            )
            indravel = np.ravel_multi_index(
                np.moveaxis(self.proj_grid_voxel_ncam[i, ..., ::-1], -1, 0),
                self.image_hw,
            )
            self.proj_grid_voxel_ncam_indravel[i] = indravel[...]

        self.grid_1d = (xgrid, ygrid, zgrid)
        self.grid_coord_3d = np.stack(
            [x_coord_3d, y_coord_3d, z_coord_3d], axis=-1
        ).astype(
            "float16"
        )  # (nvox_y, nvox_x, nvox_z, 3)

        for voxel_size in self.voxel_size_list:
            self.grid_1d_l.append(self.grid_1d)
            self.grid_coord_3d_l.append(self.grid_coord_3d)
            self.proj_grid_voxel_ncam_indravel_l.append(
                self.proj_grid_voxel_ncam_indravel
            )

    def quary_gridsample_by_com3d(
        self, com_3d: np.ndarray, ims: np.ndarray, iclass: int
    ):
        # input=gray, output=gray.
        assert len(com_3d) == 3
        assert len(ims) == self.ncam
        grid_coord_3d = self.grid_coord_3d_l[iclass]
        grid_1d = self.grid_1d_l[iclass]
        proj_grid_voxel_ncam_indravel = self.proj_grid_voxel_ncam_indravel_l[iclass]
        com_index = np.array(
            [np.searchsorted(grid_1d[i], com_3d[i], side="right") for i in range(3)]
        )  # (3,)
        com_range = np.floor(
            com_index[:, None] + [-self.nvox / 2, self.nvox / 2]
        ).astype(
            int
        )  # (3,2)

        xgrid_roi = grid_coord_3d[
            com_range[1][0] : com_range[1][1],
            com_range[0][0] : com_range[0][1],
            com_range[2][0] : com_range[2][1],
            :,
        ]  # (nvox_y, nvox_x, nvox_z, 3)

        result = voxel_sampling_c_last(
            ims[..., 0],
            proj_grid_voxel_ncam_indravel,
            com_range[[1, 0, 2]][:, 0],
            (self.nvox, self.nvox, self.nvox),
        )  # nvox, nvox, nvox, nview
        X = result.astype(np.float32)  # (nvox, nvox, nvox, nview)
        assert self.norm_im
        X = processing.preprocess_3d(X)
        y_3d = np.empty((self.nvox, self.nvox, self.nvox, 14), dtype="float32")

        return [X, xgrid_roi], y_3d


def post_cpu(
    rpc_client,
    ind_max,
    X_grid,
    iframe,
    pannel_preview,
    calibPredict: CalibPredict,
    vidout,
    timecode
):
    if iframe < 0:
        return
    com_3d_l = X_grid[:, [0, -1], [0, -1], [0, -1]].mean(axis=1)  # (nclass, 3)
    p3d = np.take_along_axis(
        X_grid.reshape(X_grid.shape[0], -1, 1, X_grid.shape[-1]),
        ind_max[:, None, :, None],
        axis=1,
    )[
        :, 0, :
    ]  # (nclass, k, 3)
    ipannel = 1
    com_2d_l = calibPredict.p3d_to_p2d(com_3d_l)[ipannel].astype(np.int32)
    p2d = calibPredict.p3d_to_p2d(p3d).astype(int)
    coord_2d = p2d[ipannel]
    pts2d_b_now, pts2d_w_now = coord_2d[0], coord_2d[1]
    frame = pannel_preview
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # b,g,r
    for ianimal in range(len(com_2d_l)):
        cv2.circle(frame, tuple(com_2d_l[ianimal]), 10, colors[ianimal], -1)

    frame = cv_plot_skeleton_aframe(frame, pts2d_b_now, name="black")
    frame = cv_plot_skeleton_aframe(frame, pts2d_w_now, name="white")
    frame = np.ascontiguousarray(frame[::2,::2])
    water_timecode(frame, timecode)

    if False:
        label_str = rpc_client.label_str()
        font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 2, 2
        text_size, _ = cv2.getTextSize(label_str, font, font_scale, font_thickness)
        image_height, image_width = frame.shape[:2]
        text_x, text_y = (
            (image_width - text_size[0]) // 2,
            (image_height - text_size[1]) - 20,
        )
        cv2.putText(
            frame,
            label_str,
            (text_x, text_y),
            font,
            font_scale,
            (0, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    vidout.write(frame)
    return p3d


def video_generator_worker(
    params: Dict,
    ba_poses: Dict,
    model_file: AnyStr,
    shared_array_imgNKHW: SynchronizedArray,
    shared_array_com2d: SynchronizedArray,
    shared_array_previ: SynchronizedArray,
    shared_array_timecode: SynchronizedArray,
    q: Queue,
    lock: Lock,
    share_dannce_X: SynchronizedArray,
    share_dannce_Xgrid: SynchronizedArray,
    share_dannce_imgpreview: SynchronizedArray,
    q_dannce: Queue,
):
    from dannce.utils_cxf.cameraIntrinsics_OpenCV import cv2_pose_to_matlab_pose
    from collections import OrderedDict

    # share array buffer
    (
        numpy_dannce_X,
        numpy_dannce_Xgrid,
        numpy_dannce_imgpreview,
    ) = get_numpy_handle_dannce(
        share_dannce_X, share_dannce_Xgrid, share_dannce_imgpreview
    )

    # init video
    params["crop_width"] = np.array(params["crop_width"]).tolist()
    params["crop_height"] = np.array(params["crop_height"]).tolist()

    # Depth disabled until next release.
    assert params["predict_mode"] == "torch"
    assert not params["expval"]

    params["depth"] = False
    n_views = int(params["n_views"])
    gpu_id = 1
    params["base_exp_folder"] = "."
    datadict = {}  # label:= filenames
    datadict_3d = {}  # 3d xyz
    com3d_dict = {}
    cameras = {}
    camnames = {}
    params["experiment"] = {}

    voxellength, coms_3d_fake = get_voxel_size(model_file)
    params["vol_size"] = voxellength
    print(f"Use real voxellength: {voxellength} mm")
    params["vmin"] = params["vmax"] = voxellength // 2

    camParams = cv2_pose_to_matlab_pose(ba_poses)
    calibPredict = CalibPredict({"ba_poses": ba_poses})
    ncamera = len(camParams)
    assert ncamera == n_views, "ncamera != n_views"
    e = 0
    camnames[e] = [f"Camera{i+1}" for i in range(ncamera)]
    cameras[e] = OrderedDict(zip(camnames[e], camParams))

    vids = None
    params["camParams"] = camParams

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
        valid_params["vol_size"] = params["vol_size"]

    # Datasets
    partition = {"valid_sampleIDs": [f"0_{i}" for i in range(180000)]}

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
        **valid_params,
    )
    numpy_imgNKHW, numpy_com2d, numpy_previ, numpy_timecode = get_numpy_handle(
        shared_array_imgNKHW,
        shared_array_com2d,
        shared_array_previ,
        shared_array_timecode,
    )
    valid_generator.set_video(
        coms_3d_fake, calibPredict, numpy_imgNKHW, numpy_com2d, numpy_previ, q, lock
    )
    for iframe in itertools.count():
        idx = iframe % NFRAME
        X, X_grid, pannel_preview = valid_generator[iframe]
        if X is None:
            break
        numpy_dannce_X[idx] = X
        numpy_dannce_Xgrid[idx] = X_grid
        numpy_dannce_imgpreview[idx] = pannel_preview
        q_dannce.put(iframe)
    q_dannce.put(None)


def dannce_predict_video_trt(
    params: Dict,
    ba_poses: Dict,
    model_file: AnyStr,
    shared_array_imgNKHW: SynchronizedArray,
    shared_array_com2d: SynchronizedArray,
    shared_array_previ: SynchronizedArray,
    shared_array_timecode: SynchronizedArray,
    q: Queue,
    lock: Lock,
):
    """Predict with dannce network

    Args:
        params (Dict): Paremeters dictionary.
    """
    from torch2trt import TRTModule
    from lilab.yolo_seg.sockerServer import port
    import picklerpc

    # Save
    _, _, _, numpy_timecode = get_numpy_handle(
        shared_array_imgNKHW,
        shared_array_com2d,
        shared_array_previ,
        shared_array_timecode,
    )
    (
        share_dannce_X,
        share_dannce_Xgrid,
        share_dannce_imgpreview,
    ) = create_shared_arrays_dannce()
    (
        numpy_dannce_X,
        numpy_dannce_Xgrid,
        numpy_dannce_imgpreview,
    ) = get_numpy_handle_dannce(
        share_dannce_X, share_dannce_Xgrid, share_dannce_imgpreview
    )

    ctx = multiprocessing.get_context("spawn")
    rpc_client = picklerpc.Client(("127.0.0.1", port))

    if True:  # 展示关闭
        q_p3d = ctx.Queue(maxsize=(NFRAME - 4))
        process = ctx.Process(target=cluster_main, args=(q_p3d, model_file))
        process.start()

    # Create a process for data generator
    q_dannce = ctx.Queue(maxsize=(NFRAME - 4))
    process = ctx.Process(
        target=video_generator_worker,
        args=(
            params,
            ba_poses,
            model_file,
            shared_array_imgNKHW,
            shared_array_com2d,
            shared_array_previ,
            shared_array_timecode,
            q,
            lock,
            share_dannce_X,
            share_dannce_Xgrid,
            share_dannce_imgpreview,
            q_dannce,
        ),
    )
    process.start()

    # Load model from tensorrt
    assert params["dannce_predict_model"] is not None
    mdl_file = params["dannce_predict_model"].replace(".hdf5", ".idx.engine")
    print("Loading model from " + mdl_file)
    assert os.path.exists(mdl_file), f"Model file {mdl_file} not found"
    calibPredict = CalibPredict({"ba_poses": ba_poses})

    # create and warm up models
    model_l = []
    device_l = ["cuda:1", "cuda:2"]
    for device in device_l:
        with torch.cuda.device(device):
            model = TRTModule()
            model.load_from_engine(mdl_file)
            model_l.append(model)

            # warm up
            idx = model.engine.get_binding_index(model.input_names[0])
            shape = tuple(model.context.get_binding_shape(idx))
            if shape[0] == -1:
                shape = (2, *shape[1:])
            input = torch.empty(size=shape).cuda()
            output = model(input)

    vidout = get_vidout()

    from lilab.timecode_tag.netcoder import Netcoder

    nettimecoder = Netcoder()
    iter_process = tqdm.tqdm(itertools.count(), desc="3D_poses", position=1)

    for iframe in iter_process:
        idx = q_dannce.get()
        if idx is None:
            break
        idx = idx % NFRAME
        timecode_delays = numpy_timecode[idx]
        timecode = timecode_delays[0]
        dt1 = nettimecoder.getTimeDelay(timecode)
        X = numpy_dannce_X[idx]
        X_grid = numpy_dannce_Xgrid[idx]
        pannel_preview = numpy_dannce_imgpreview[idx]

        pred_l = [None] * len(device_l)
        for i, (device, model) in enumerate(zip(device_l, model_l)):
            with torch.cuda.device(device):
                pred_l[i] = model(torch.from_numpy(X[[i]]).cuda().float())

        for device in device_l:
            with torch.cuda.device(device):
                torch.cuda.current_stream().synchronize()
        ind_max = np.concatenate([pred.cpu().numpy() for pred in pred_l])
        p3d = post_cpu(
            rpc_client, ind_max, X_grid, iframe, pannel_preview, calibPredict, vidout, timecode
        )
        q_p3d.put((p3d, timecode))

        dt2 = nettimecoder.getTimeDelay(timecode)
        dt_str = str(int(dt2)) if not np.isnan(dt2) else "x"
        numpy_timecode[idx, [3, 4]] = [dt1, dt2]
        iter_process.set_description(
            "[3D Pose] q={:>2}, i={:>4}, delay={:>3}".format(
                q_dannce.qsize(), idx, dt_str
            )
        )

    raise "Nerver reach here"
    q_p3d.put(None)
    print("[2] Pose reconstruction done")
    vidout.release()
