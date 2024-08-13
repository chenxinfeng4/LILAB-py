# conda activate mmdet; python /home/liying_lab/chenxf/ml-project/LILAB-py/lilab/yolo_seg/fun_main.py
import numpy as np
import multiprocessing
from lilab.label_live.t1_realtime_position import main as seg_main
from lilab.yolo_seg.common_variable import (
    NFRAME, out_numpy_imgNKHW_shape, out_numpy_com2d_shape,
    out_numpy_previ_shape, out_numpy_timecode_shape)

CALIBPKL = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/test/2022-10-13_15-08-49AWxCB.maskrcnn.matcalibpkl'
dannce_project ='/home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x9_rat_yolo_metric'
model_smooth_matcalibpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/test/2022-10-13_15-08-49AWxCB.maskrcnn.matcalibpkl'


if __name__ == '__main__':
    ctx = multiprocessing.get_context('spawn')
    q = ctx.Queue(maxsize=(NFRAME-4))
    shared_array_imgNNHW = multiprocessing.Array('b', int(NFRAME*np.prod(out_numpy_imgNKHW_shape)))  #np.uint8
    shared_array_com2d = multiprocessing.Array('d', int(NFRAME*np.prod(out_numpy_com2d_shape)))      #np.float64
    shared_array_previ = multiprocessing.Array('b', int(NFRAME*np.prod(out_numpy_previ_shape)))      #np.uint8
    shared_array_timecode = multiprocessing.Array('d', int(NFRAME*out_numpy_timecode_shape)) #np.float64

    process = ctx.Process(target=seg_main, args=(shared_array_imgNNHW, shared_array_com2d, shared_array_previ, shared_array_timecode, q))
    process.start()
    
    while True:
        q.get()
