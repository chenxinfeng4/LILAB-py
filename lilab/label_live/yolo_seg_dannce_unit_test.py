# conda activate mmdet; python /home/liying_lab/chenxf/ml-project/LILAB-py/lilab/yolo_seg/fun_main.py
import os
import os.path as osp
import numpy as np
import multiprocessing
import pickle
from lilab.label_live.t1_realtime_position import main as seg_main
from lilab.label_live.plugin_voxelpred import dannce_predict_video_trt as dannce_main
from lilab.yolo_seg.sockerServer import start_socketserver_background
from lilab.yolo_seg.common_variable import (
    NFRAME, create_shared_arrays)

from dannce import _param_defaults_dannce, _param_defaults_shared
from dannce.interface_cxf import build_params
from dannce.engine import processing_cxf as processing

CALIBPKL = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/test/2022-10-13_15-08-49AWxCB.maskrcnn.matcalibpkl'
dannce_project ='/home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x9_rat_yolo_metric'
model_smooth_matcalibpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/test/2022-10-13_15-08-49AWxCB.maskrcnn.matcalibpkl'


if __name__ == '__main__':
    ctx = multiprocessing.get_context('spawn')
    q = ctx.Queue(maxsize=(NFRAME-4))
    shared_array_imgNNHW, shared_array_com2d, shared_array_previ, shared_array_timecode = create_shared_arrays()
    process = ctx.Process(target=seg_main, args=(shared_array_imgNNHW, shared_array_com2d, shared_array_previ, shared_array_timecode, q))
    process.start()

    
    lock = ctx.Lock()
    start_socketserver_background()

    os.chdir(dannce_project)
    params = {**_param_defaults_dannce, **_param_defaults_shared}
    params_file = build_params(osp.join(dannce_project, '../../configs/dannce_rat14_1280x800x9_max_config.yaml'), dannce_net=True)
    params.update(params_file)
    params = processing.infer_params_img(params, dannce_net=True, prediction=True)
    ba_poses = pickle.load(open(CALIBPKL, 'rb'))['ba_poses']

    dannce_main(params, ba_poses,
                model_smooth_matcalibpkl,
                shared_array_imgNNHW,
                shared_array_com2d,
                shared_array_previ,
                shared_array_timecode,
                q, lock)
