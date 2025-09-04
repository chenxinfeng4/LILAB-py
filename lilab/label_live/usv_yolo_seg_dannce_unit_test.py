# conda activate mmdet; python /home/liying_lab/chenxf/ml-project/LILAB-py/lilab/label_live/usv_yolo_seg_dannce_unit_test.py
import os
import os.path as osp
import numpy as np
import time
import multiprocessing
import pickle
from lilab.label_live.t1_realtime_position import main as seg_main
from lilab.label_live.plugin_voxelpred import dannce_predict_video_trt as dannce_main
# from lilab.label_live.sockerServer import start_socketserver_background
from lilab.yolo_seg.common_variable import NFRAME, create_shared_arrays
from dannce import _param_defaults_dannce, _param_defaults_shared
from dannce.interface_cxf import build_params
from dannce.engine import processing_cxf as processing

# CALIBPKL = "/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/test/2022-10-13_15-08-49AWxCB.maskrcnn.matcalibpkl"
# model_smooth_matcalibpkl = "/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/test/2022-10-13_15-08-49AWxCB.maskrcnn.matcalibpkl"
dannce_project = (
    "/home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x9_rat_yolo_metric"
)

# CALIBPKL = "/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day55_Mix_analysis/Day55_yolo_rt/2023-05-24_14-43-58CBxDW.matcalibpkl"
# CALIBPKL = "/mnt/liying.cibr.ac.cn_Data_Temp_ZZC_bak/2024-ChR2rat/2409s3test/2024-09-08_18-37-32_RatPlaySample.smoothed_foot.matcalibpkl"
# CALIBPKL = '/mnt/liying.cibr.ac.cn_Data_Temp_bak/carl/2024-12-29_12-24-17.calibpkl'
CALIBPKL = '/mnt/liying.cibr.ac.cn_Data_Temp_ZZC2/2506batch-chr2/test/ball/2025-06-13_10-16-58_ball.calibpkl'  #ana 20250428
model_smooth_matcalibpkl = '/DATA/zhongzhenchao/2501chr2-shank3/all400p/2025-01-05_15-04-37_l7_sm1_pm6.smoothed_foot.matcalibpkl'



if __name__ == "__main__":
    ctx = multiprocessing.get_context("spawn")
    lock = ctx.Lock()
    # start_socketserver_background()

    q = ctx.Queue(maxsize=(NFRAME - 2))
    (
        shared_array_imgNNHW,
        shared_array_com2d,
        shared_array_previ,
        shared_array_timecode,
    ) = create_shared_arrays()
    process = ctx.Process(
        target=seg_main,
        args=(
            shared_array_imgNNHW,
            shared_array_com2d,
            shared_array_previ,
            shared_array_timecode,
            q,
        ),
    )
    process.start()

    # process_usv = ctx.Process(target=usv_main)
    # process_usv.start()

    os.chdir(dannce_project)
    params = {**_param_defaults_dannce, **_param_defaults_shared}
    params_file = build_params(
        osp.join(
            dannce_project, "../../configs/dannce_rat14_1280x800x9_max_config.yaml"
        ),
        dannce_net=True,
    )
    params.update(params_file)
    params = processing.infer_params_img(params, dannce_net=True, prediction=True)
    ba_poses = pickle.load(open(CALIBPKL, "rb"))["ba_poses"]

    dannce_main(
        params,
        ba_poses,
        model_smooth_matcalibpkl,
        shared_array_imgNNHW,
        shared_array_com2d,
        shared_array_previ,
        shared_array_timecode,
        q,
        lock,
    )
