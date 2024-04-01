# python -m lilab.yolo_det.s1_video2matpkl /mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-1-16/ball_move_cam1.mp4 --pannels 4
# %%
import argparse
import numpy as np
import torch
import tqdm
from lilab.yolo_det.convert_pt2onnx import singleton
from lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_realtimecam import (
    transform_preds, box2cs, mid_gpu
)
import os
import os.path as osp
from lilabnext.multiview_zyy.video_set_reader import VideoSetReader
from torch2trt import TRTModule
import warnings
import pickle
import ffmpegcv
from ffmpegcv.ffmpeg_reader_pannels import FFmpegReaderPannels
from lilab.cameras_setup import get_view_xywh_wrapper

checkpoint = '/home/liying_lab/chenxinfeng/DATA/ultralytics/work_dirs/yolov8_det_640x640_marmoset/weights/last.singleton.engine'

def post_cpu(outputs, center, scale, feature_in_wh):
    boxes, scores = singleton(outputs)
    boxes_center = (boxes[...,[0,1]] + boxes[...,[2,3]])/2
    [W, H] = feature_in_wh
    preds = transform_preds(boxes_center, center, scale, [W, H], use_udp=False)
    keypoints_xyp = np.concatenate((preds, scores[...,None]), axis=-1) #(N, K, xyp)
    return keypoints_xyp


class DataSet: 
    def __init__(self, vid:VideoSetReader):
        print('vid.out_numpy_shape', vid.out_numpy_shape)
        self.coord_NCHW_idx_ravel = np.zeros(vid.out_numpy_shape, dtype=np.uint8).transpose(0,3,1,2)
        self.vid = vid
        self.input_trt_shape = self.coord_NCHW_idx_ravel.shape
    
    def __len__(self):
        return len(self.vid)
        
    def __iter__(self):
        while True:
            ret, img_NHWC = self.vid.read()
            if not isinstance(img_NHWC, np.ndarray):
                img_NHWC = np.stack(img_NHWC, axis=0)
            if not ret:
                print('End of video')
                raise StopIteration
            img_preview = None
            img_NCHW = img_NHWC.transpose(0,3,1,2).astype(np.float32)
            yield img_NCHW, img_preview

class DataSet0:
    def __init__(self, vid):
        [N,H,W,C] = vid.out_numpy_shape
        print('vid.out_numpy_shape', vid.out_numpy_shape)
        self.coord_NCHW_idx_ravel = np.zeros([N,C,H,W], dtype=np.uint8)
        self.vid = vid
        self.input_trt_shape = self.coord_NCHW_idx_ravel.shape

    def __len__(self):
        return len(self.vid)
    
    def __iter__(self):
        while True:
            ret, img_NHWC = self.vid.read()
            if not ret:
                print('End of video')
                raise StopIteration
            img_preview = None
            img_NCHW = img_NHWC.transpose(0,3,1,2).astype(np.float32)
            yield img_NCHW, img_preview


def create_trtmodule(checkpoint, input_shape0):
    trt_model = TRTModule()
    trt_model.load_from_engine(checkpoint)
    idx = trt_model.engine.get_binding_index(trt_model.input_names[0])
    input_shape = tuple(trt_model.context.get_binding_shape(idx))
    print('input_shape0', input_shape0, input_shape)
    if input_shape[0]==-1:
        assert input_shape[1:]==input_shape0[1:]
        input_shape = input_shape0
    else:
        assert input_shape==input_shape0
    img_NCHW = np.ones(input_shape)
    outputs = mid_gpu(trt_model, img_NCHW)
    return trt_model, img_NCHW, outputs


def main(video_file, checkpoint, setupname):
    views_xywh = get_view_xywh_wrapper(setupname)
    nview = len(views_xywh)
    if '_cam' in osp.splitext(osp.basename(video_file))[0]:
        feature_in_wh = [640, 480]
        vid = VideoSetReader(video_file, nvideo=nview, pix_fmt='rgb24',
                            resize = feature_in_wh, resize_keepratio=False)
        view_w, view_h = vid.vid_list[0].origin_width, vid.vid_list[0].origin_height
        dataset = DataSet(vid)
        print('view_w, h', view_w, view_h)
    else:
        vid = ffmpegcv.VideoCapturePannels(video_file, pix_fmt='rgb24', crop_xywh_l=views_xywh)
        view_w, view_h = feature_in_wh = views_xywh[0][2:]
        dataset = DataSet0(vid)
        print('view_w, h', view_w, view_h)
    
    center, scale = box2cs(np.array([0,0,view_w,view_h]), feature_in_wh, keep_ratio=False)
    dataset_iter = iter(dataset)
    
    input_shape = dataset.coord_NCHW_idx_ravel.shape
    trt_model, img_NCHW, outputs = create_trtmodule(checkpoint, input_shape)

    keypoints_xyp = []
    for idx in tqdm.trange(-1, len(dataset)-1):
        outputs_wait = mid_gpu(trt_model, img_NCHW)              # t
        img_NCHW_next, img_preview_next = next(dataset_iter)     # t+1
        img_NCHW, img_preview = img_NCHW_next, img_preview_next  # t+1
        torch.cuda.current_stream().synchronize()                # t
        outputs = outputs_wait                                   # t
        if idx<=-1: continue
        kpt2d = post_cpu(outputs, center, scale, feature_in_wh)  # t
        keypoints_xyp.append(kpt2d)

    outputs = mid_gpu(trt_model, img_NCHW)
    kpt2d = post_cpu(outputs, center, scale, feature_in_wh)
    keypoints_xyp.append(kpt2d)

    keypoints_xyp = np.array(keypoints_xyp).transpose(1,0,2,3)#(nview, T, K, 3)
    # assert np.mean(keypoints_xyp[...,2].ravel()<0.4) < 0.1, 'Too many nan in keypoints_xyp !'
    if np.median(keypoints_xyp[...,2])<0.4:
        warnings.warn('Too many nan in keypoints_xyp !')
        # assert np.median(keypoints_xyp[...,2])>0.4, 'Too many nan in keypoints_xyp !'

    info = {'vfile': video_file, 'nview': nview, 'fps':  dataset.vid.fps}
    outpkl = os.path.splitext(video_file)[0] + '.matpkl'
    outdict = dict(
        keypoints = keypoints_xyp,
        views_xywh = views_xywh,
        info = info
    )
    pickle.dump(outdict, open(outpkl, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, help='path to video or folder')
    parser.add_argument('--setupname', default='frank', type=str, help='number views')
    parser.add_argument('--checkpoint', type=str, default=checkpoint)
    arg = parser.parse_args()

    video_path, checkpoint = arg.video_path, arg.checkpoint
    print("checkpoint:", checkpoint)
    assert osp.isfile(video_path), 'video_path not exists'
    main(video_path, checkpoint, arg.setupname)
