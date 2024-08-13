import cv2
import torch
import tqdm
import ffmpegcv
from torch2trt import TRTModule
import numpy as np
from lilab.yolo_seg.utilities import *
from itertools import product
from lilab.cameras_setup import get_view_xywh_wrapper
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing import Queue
from lilab.yolo_seg.common_variable import (
    NFRAME, out_numpy_imgNKHW_shape, out_numpy_com2d_shape, 
    out_numpy_previ_shape, out_numpy_timecode_shape)

from ffmpegcv.ffmpeg_noblock import ReadLiveLast
from lilab.timecode_tag.decoder import getDecoder
from lilab.timecode_tag.netcoder import getTimeDelay
#segdate = NVIEW,NFRAME,[box|seg],NCLASS,NINSTANCE=1

engine='/home/liying_lab/chenxinfeng/DATA/ultralytics/work_dirs/yolov8n_seg_640_ratbw_extra/weights/last.full.engine'  #train3

# vid = ReadLiveLast(ffmpegcv.VideoCaptureStreamRT, 'rtsp://10.50.60.6:8554/mystream', pix_fmt='gray')
# vid = ffmpegcv.VideoCaptureStreamRT('rtmp://10.50.5.83:1935/mystream', pix_fmt='gray')
# ret, frame = vid.read()
# vid.count = 400000


# vid.ffmpeg_cmd = vid.ffmpeg_cmd.replace('ffmpeg ', 'ffmpeg -re ')
def get_vidin():
    # vid = ffmpegcv.VideoCaptureNV('/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/test/2022-10-13_15-08-49AWxCB_5min.mp4',
    #                         pix_fmt='gray')
    vid = ReadLiveLast(ffmpegcv.VideoCaptureStreamRT, 'rtsp://10.50.60.6:8554/mystream_9cam', pix_fmt='gray')
    vidout = ffmpegcv.VideoWriterStreamRT('rtsp://10.50.60.6:8554/mystream_preview', pix_fmt='gray')
    return vid, vidout


pannelWH = (1280, 800)


def mid_gpu(trt_model, img_NCHW:np.ndarray):
    if isinstance(img_NCHW, np.ndarray):
        img_NCHW = torch.from_numpy(img_NCHW).cuda().float()
    return trt_model(img_NCHW)


def main(shared_array_imgNNHW:SynchronizedArray,
         shared_array_com2d:SynchronizedArray,
         shared_array_previ:SynchronizedArray,
         shared_array_timecode:SynchronizedArray,
         q:Queue):
    numpy_imgNKHW = np.frombuffer(shared_array_imgNNHW.get_obj(), dtype=np.uint8).reshape((NFRAME,*out_numpy_imgNKHW_shape))
    numpy_com2d = np.frombuffer(shared_array_com2d.get_obj(), dtype=np.float64).reshape((NFRAME, *out_numpy_com2d_shape))
    numpy_previ = np.frombuffer(shared_array_previ.get_obj(), dtype=np.uint8).reshape((NFRAME, *out_numpy_previ_shape))
    numpy_timecode = np.frombuffer(shared_array_timecode.get_obj(), dtype=np.float64).reshape((NFRAME, out_numpy_timecode_shape)) #(timecode, delay1, delay2, delay3)

    vid, vidout = get_vidin()
    assert (vid.width, vid.height) == (pannelWH[0]*3, pannelWH[1]*3)

    with torch.cuda.device('cuda:0'):
        timecode_decoder = getDecoder()
        trt_model = TRTModule()
        trt_model.load_from_engine(engine)
        self_id = 0
        #检查 trt 模型，获取输入输出尺寸
        input_shape = tuple(trt_model.context.get_binding_shape(0))
        assert input_shape == (vid.height//2, vid.width//2)
        singleton_gpu = singleton_gpu_factory(trt_model)

        if True:
            img_H0W0 = np.random.rand(*input_shape).astype(np.float32)
            outputs = mid_gpu(trt_model,img_H0W0)
            torch.cuda.current_stream().synchronize()
            boxes_, scores_, mask_ = singleton_gpu(outputs)
            scores, box_for_mask, mask_within_roi, coms_real_2d = refine_mask((boxes_, scores_, mask_))

        count_range = range(len(vid)) if hasattr(vid, 'count') and vid.count else range(400000)
        nview, nclass = scores.shape
        crop_xy = np.array(get_view_xywh_wrapper('carl'))[:,:2]  #(nview,2)
        crop_xy_ = np.concatenate([crop_xy, crop_xy], axis=-1)  #(nview,4)

        def pre_cpu(vid, vidout):
            ret, frame = vid.read()
            if not ret: exit(0)
            frame = np.squeeze(frame)
            timecode, *_ = timecode_decoder(frame)
            frame_small = np.ascontiguousarray(frame[::2,::2])
            img_H0W0 = frame_small.reshape(*input_shape)
            frame_preview2 = np.ascontiguousarray(frame[0:pannelWH[1]:2, 0:pannelWH[0]*2:2])
            # vidout.write(frame_preview2)
            return timecode, frame, img_H0W0
        
        # outvid = ffmpegcv.VideoWriter(outvfile, None, 30)
        def post_cpu(frame, outputs, queue_idx:int):
            idx = queue_idx % NFRAME
            canvas = numpy_imgNKHW[idx]  #canvas: masked image from multi cameras, (nview, nanimal, H, W)
            assert canvas.shape == (nview, nclass, *pannelWH[::-1])
            canvas[:] = 0
            coms_real_2d = numpy_com2d[idx]
            numpy_previ[idx] = cv2.cvtColor(np.ascontiguousarray(
                frame[:pannelWH[1], pannelWH[0]:pannelWH[0]*2]),cv2.COLOR_RGB2BGR)

            boxes_, scores_, mask_ = singleton_gpu(outputs)
            scores, box_for_mask_orign, mask_orign_within_roi, coms_real_2d[:] = refine_mask((boxes_, scores_, mask_))

            box_for_frame_restore = box_for_mask_orign + crop_xy_[:,None,:] #(nview, nclass, 4)
            for iview, iclass in product(range(nview), range(nclass)):
                bx, by, bx2, by2 = box_for_mask_orign[iview, iclass, :]
                fx, fy, fx2, fy2 = box_for_frame_restore[iview, iclass, :]
                score = scores[iview, iclass]
                if score<=0: continue
                canvas[iview, iclass, by:by2, bx:bx2] = frame[fy:fy2, fx:fx2] * mask_orign_within_roi[iview, iclass]

            q.put(idx)
            return idx, canvas, coms_real_2d

        iter_process = tqdm.tqdm(count_range, 
                                        desc='worker[{}]'.format(self_id),
                                        position=int(self_id))
        for idx, _ in enumerate(iter_process):
            iter_process.set_description('q={}, i={}'.format(q.qsize(), idx%100))
            timecode, frame, img_H0W0 = pre_cpu(vid, vidout)        # t
            dt1 = getTimeDelay(timecode)
            outputs = mid_gpu(trt_model,img_H0W0)     # t
            buf_idx, canvas, coms_real_2d = post_cpu(frame, outputs, queue_idx=idx)
            dt2 = getTimeDelay(timecode)
            numpy_timecode[buf_idx,:3] = [timecode, dt1, dt2]

        q.put(None)
        print('[1] Video masking worker done!')
