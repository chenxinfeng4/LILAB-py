# python -m lilab.yolo_det.s1_video2matpkl /mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-1-16/ball_move_cam1.mp4 --pannels 4
# %%
import argparse
import numpy as np
import torch
import tqdm
from lilab.yolo_det.convert_pt2onnx_nms import nms
from lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_realtimecam import (
    transform_preds, box2cs, mid_gpu
)
import os
import os.path as osp
from lilabnext.multiview_zyy.video_set_reader import VideoSetReader
import warnings
import pickle
import ffmpegcv
from lilab.cameras_setup import get_view_xywh_wrapper
from lilab.yolo_det.s1_video2matpkl import (
    DataSet, DataSet0, create_trtmodule)

checkpoint = '/home/liying_lab/chenxinfeng/DATA/ultralytics/work_dirs/yolov8_det_640x640_marmoset/weights/last.singleton.engine'

iclass = 0
nins = 2

#%%
def post_cpu(outputs, center, scale, feature_in_wh, iclass, nins):
    boxes, scores, nums = nms(outputs)
    boxes, scores, nums = boxes[:,iclass], scores[:,iclass], nums[:,iclass]
    nview = len(boxes)
    boxes_full = np.zeros((nview, nins, 4))+np.nan
    scores_full = np.zeros((nview, nins))
    for i in range(nview):
        nums_i = min(nins, nums[i])
        boxes_full[i][:nums_i] = boxes[i][:nums_i]
        scores_full[i][:nums_i] = scores[i][:nums_i]
    boxes, scores = boxes_full, scores_full
    boxes_center = (boxes[...,[0,1]] + boxes[...,[2,3]])/2
    [W, H] = feature_in_wh
    preds = transform_preds(boxes_center, center, scale, [W, H], use_udp=False)
    keypoints_xyp = np.concatenate((preds, scores[...,None]), axis=-1) #(N, K, xyp)
    return keypoints_xyp


def main(video_file, checkpoint, setupname, iclass, nins):
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
        kpt2d = post_cpu(outputs, center, scale, feature_in_wh, iclass, nins)  # t
        keypoints_xyp.append(kpt2d)

    outputs = mid_gpu(trt_model, img_NCHW)
    kpt2d = post_cpu(outputs, center, scale, feature_in_wh, iclass, nins)
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
    parser.add_argument('--ninstance', type=int, default=2)
    parser.add_argument('--iclass', type=int, default=0)
    arg = parser.parse_args()

    video_path, checkpoint = arg.video_path, arg.checkpoint
    print("checkpoint:", checkpoint)
    assert osp.isfile(video_path), 'video_path not exists'
    main(video_path, checkpoint, arg.setupname, arg.iclass, arg.ninstance)
