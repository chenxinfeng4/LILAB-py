import cv2
import torch
import tqdm
import ffmpegcv
from torch2trt import TRTModule
import numpy as np
from lilab.yolo_seg.utilities import *
from itertools import product
from typing import Tuple
from lilab.cameras_setup import get_view_xywh_wrapper
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing import Queue
from multiprocessing.synchronize import Lock
from lilab.yolo_seg.common_variable import (
    NFRAME, out_numpy_imgNKHW_shape, out_numpy_com2d_shape, out_numpy_previ_shape)

from ffmpegcv.ffmpeg_noblock import ReadLiveLast
#segdate = NVIEW,NFRAME,[box|seg],NCLASS,NINSTANCE=1

nclass = 2
engine='/home/liying_lab/chenxinfeng/DATA/ultralytics/work_dirs/yolov8n_seg_640_ratbw/weights/last.full.engine'  #train3
outvfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/seg_com2d.mkv'
# ordered_trtoutputname = ['bboxes','scores','maskcoeff','proto']
ordered_trtoutputname = ['maskcoeff','bboxes','scores','proto']
# vid = ReadLiveLast(ffmpegcv.VideoCaptureStreamRT, 'rtsp://10.50.60.6:8554/mystream', pix_fmt='gray')
# vid = ffmpegcv.VideoCaptureStreamRT('rtmp://10.50.5.83:1935/mystream', pix_fmt='gray')
# ret, frame = vid.read()
# vid.count = 400000

vid = ffmpegcv.VideoCaptureNV('/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/test/2022-10-13_15-08-49AWxCB.mp4',
                            pix_fmt='gray')

# vid.ffmpeg_cmd = vid.ffmpeg_cmd.replace('ffmpeg ', 'ffmpeg -re ')
pannelWH = (1280, 800)
assert (vid.width, vid.height) == (pannelWH[0]*3, pannelWH[1]*3)

def mid_gpu(trt_model, img_NCHW:np.ndarray):
    if isinstance(img_NCHW, np.ndarray):
        img_NCHW = torch.from_numpy(img_NCHW).cuda().float()
    return trt_model(img_NCHW)


def main(shared_array_imgNNHW:SynchronizedArray,
         shared_array_com2d:SynchronizedArray,
         shared_array_previ:SynchronizedArray,
         q:Queue, lock:Lock):
    numpy_imgNNHW = np.frombuffer(shared_array_imgNNHW.get_obj(), dtype=np.uint8).reshape((NFRAME,*out_numpy_imgNKHW_shape))
    numpy_com2d = np.frombuffer(shared_array_com2d.get_obj(), dtype=np.float64).reshape((NFRAME, *out_numpy_com2d_shape))
    numpy_previ = np.frombuffer(shared_array_previ.get_obj(), dtype=np.uint8).reshape((NFRAME, *out_numpy_previ_shape))

    with torch.cuda.device('cuda:0'):
        trt_model = TRTModule()
        trt_model.load_from_engine(engine)
        self_id = 0
        #检查 trt 模型，获取输入输出尺寸
        input_shape = tuple(trt_model.context.get_binding_shape(0))
        assert input_shape == (vid.height//2, vid.width//2)
        trt_output_names = trt_model.output_names
        assert set(trt_output_names) == set(ordered_trtoutputname)
        trt_output_names_to_order = [trt_output_names.index(name) for name in ordered_trtoutputname]

        if True:
            img_H0W0 = np.random.rand(*input_shape).astype(np.float32)
            outputs = mid_gpu(trt_model,img_H0W0)
            torch.cuda.current_stream().synchronize()
            outputs_ = [outputs[i] for i in trt_output_names_to_order]
            boxes_, scores_, mask_ = singleton_gpu(outputs_)
            scores, box_for_mask, mask_within_roi, coms_real_2d = refine_mask((boxes_, scores_, mask_))

        count_range = range(len(vid)) if hasattr(vid, 'count') and vid.count else range(400000)
        nview, nclass = scores.shape
        crop_xy = np.array(get_view_xywh_wrapper('carl'))[:,:2]  #(nview,2)

        crop_xy_ = np.concatenate([crop_xy, crop_xy], axis=-1)  #(nview,4)

        # outvid = ffmpegcv.VideoWriter(outvfile, None, 30)
        def post_cpu(frame, outputs):
            outputs_ = [outputs[i] for i in trt_output_names_to_order]
            boxes_, scores_, mask_ = singleton_gpu(outputs_)
            scores, box_for_mask_orign, mask_orign_within_roi, coms_real_2d = refine_mask((boxes_, scores_, mask_))

            coms_real_2d = coms_real_2d.astype(float)
            box_for_frame_restore = box_for_mask_orign + crop_xy_[:,None,:] #(nview, nclass, 4)
            canvas = np.zeros((nview, nclass, *pannelWH[::-1]), dtype=np.uint8)
            for iview, iclass in product(range(nview), range(nclass)):
                bx, by, bx2, by2 = box_for_mask_orign[iview, iclass, :]
                fx, fy, fx2, fy2 = box_for_frame_restore[iview, iclass, :]
                score = scores[iview, iclass]
                if score<=0: continue
                canvas[iview, iclass, by:by2, bx:bx2] = frame[fy:fy2, fx:fx2] * mask_orign_within_roi[iview, iclass]
            return canvas, coms_real_2d

        def pre_cpu(vid):
            ret, frame = vid.read()
            if not ret: exit(0)
            frame = np.squeeze(frame)
            frame_small = np.ascontiguousarray(frame[::2,::2])
            img_H0W0 = frame_small.reshape(*input_shape)
            return frame, img_H0W0
        
        def push_queue(idx, img_NNHW, coms_real_2d, frame_HW):
            idx = idx % NFRAME
            # with lock:
            numpy_imgNNHW[idx] = img_NNHW
            numpy_com2d[idx] = coms_real_2d
            numpy_previ[idx] = frame_HW[:800,1280:1280*2]
            if q.full(): q.get()
            # frame_HWC = cv2.cvtColor(numpy_previ[idx], cv2.COLOR_BGR2RGB)
            # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] #b,g,r
            # for ianimal, (x,y) in enumerate(coms_real_2d[1]):
            #     if np.isnan(x) or np.isnan(y): continue
            #     cv2.circle(frame_HWC, (int(x), int(y)), 5, colors[ianimal], -1)
            # outvid.write(frame_HWC)
            q.put(idx)

        iter_process = tqdm.tqdm(count_range, 
                                        desc='worker[{}]'.format(self_id),
                                        position=int(self_id))
        # for _ in range(1000): ret, frame = vid.read()
        for idx, _ in enumerate(iter_process):
            iter_process.set_description('q={}, i={}'.format(q.qsize(), idx%100))
            frame, img_H0W0 = pre_cpu(vid)            # t
            outputs = mid_gpu(trt_model,img_H0W0)     # t
            canvas, coms_real_2d = post_cpu(frame, outputs)

            # canvas_pre, coms_real_2d_pre = post_cpu(frame_pre, outputs_pre) # t-1
            push_queue(idx, canvas, coms_real_2d, frame)
            # torch.cuda.current_stream().synchronize()
