# python -m lilab.mmdet_dev.s1_mmdet_videos2pkl_trt  A.mp4 --pannels 9 [CONFIG] [CHECKPOINT]
# python -m lilab.mmdet_single.s1_mmdet_videos2pkl_single  A/B/C

import argparse
import glob
import itertools
import os
import os.path as osp
import pickle
import cv2

import ffmpegcv
import numpy as np
import torch
import tqdm
from torchvision.transforms import functional as F
from ffmpegcv.video_info import get_info
import re
# from mmdet.models.roi_heads.mask_heads import FCNMaskHead
import lilab.cvutils.map_multiprocess_cuda as mmap_cuda
from lilab.cameras_setup import get_view_xywh_wrapper
import itertools
from lilab.mmdet_dev.filter_vname import filter_vname
from torch2trt import TRTModule
from lilab.yolo_seg.utilities import (singleton_gpu_factory, refine_mask2, refine_mask3)
# from lilab.mmpose_dev.a2_convert_mmpose2engine import findcheckpoint_trt

video_path = [
    f
    for f in glob.glob(
        "/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/test/*.mp4"
    )
    if f[-4] not in "0123456789"
]


def load_trtmodel(checkpoint, cuda:int):
    with torch.cuda.device(f'cuda:{cuda}'):
        trt_model = TRTModule()
        trt_model.load_from_engine(checkpoint)
        idx = trt_model.engine.get_binding_index(trt_model.input_names[0])
        input_shape = tuple(trt_model.context.get_binding_shape(idx))
        return trt_model, input_shape


class MyWorker(mmap_cuda.Worker):
# class MyWorker():
    def __init__(self, checkpoint, maxlen):
        super().__init__()
        self.cuda = getattr(self, "cuda", 0)
        self.id = getattr(self, "id", 0)
        self.checkpoint = checkpoint
        self.maxlen = maxlen
        print("well setup worker:", self.cuda)
        self.cworker_per_gpu = 2

    def compute(self, args):
        video = args
        if osp.exists(osp.splitext(video)[0] + '.segpkl'): print('Skip:', video);return
        model, input_shape = load_trtmodel(self.checkpoint, self.cuda)
        resize_WH = input_shape[::-1]
        vid = ffmpegcv.VideoCaptureNV(video, resize=resize_WH, resize_keepratio=False,
                                      gpu=int(self.cuda),pix_fmt="gray")
        singleton_gpu = singleton_gpu_factory(model)

        maxlen = min([len(vid), self.maxlen]) if self.maxlen else len(vid)

        with torch.cuda.device(self.cuda):
            # warm up
            frame = np.zeros((resize_WH[1],resize_WH[0]),dtype=np.uint8)
            frame_cuda = torch.from_numpy(frame).cuda().float()
            gpu_outputs = model(frame_cuda)
            outputs = singleton_gpu(gpu_outputs)
            # result = refine_mask2(outputs)

        result_l = []
        with torch.cuda.device(self.cuda), torch.no_grad(), vid:
            for iframe in tqdm.trange(maxlen, position=self.id, desc=f"[{self.id}]"):
                ret, frame = vid.read() #0:n
                frame = frame.squeeze()
                # frame_cuda = torch.from_numpy(frame).cuda().float()
                frame_cuda = torch.from_numpy(frame).cuda().float()
                outputs = singleton_gpu(gpu_outputs) #-1:n-1
                gpu_outputs = model(frame_cuda) #0:n 
                if iframe>0:
                    result = refine_mask2(outputs) #-1:n-1
                    result_l.append(result)
                # #     q.put((iframe-1, result)) #-1:n-1

            outputs = singleton_gpu(gpu_outputs)
            result = refine_mask2(outputs)
            result_l.append(result)
            # # q.put((iframe, result))
        vid.release()
        convert(video, result_l)


def convert(vfile, result_l, setupname='carl'):
    views_xywh = get_view_xywh_wrapper(setupname)
    nview = len(views_xywh)
    vinfo = get_info(vfile)
    assert len(result_l[0][0]) == nview
    nframe = len(result_l)
    com2d = np.array([r[-1] for r in result_l]).transpose(1, 0, 2, 3)
    segdata = np.empty((nview, nframe, 2),dtype=object).tolist()

    for iview, iframe in itertools.product(range(nview), range(nframe)):
        box_xyxyp_ii = result_l[iframe][0][iview] #(nclass, 5)
        maskenc_ii = result_l[iframe][1][iview] #(nclass,)
        box_xyxyp_re = [([x] if x[-1]>0 else []) for x in box_xyxyp_ii]
        maskenc_re = [[x] for x in maskenc_ii]
        segdata[iview][iframe] = [box_xyxyp_re, maskenc_re]

    _, _, nclass, _ = com2d.shape
    outdata = { 'info': {
        'vfile': vfile, 
        'nview': nview, 
        'fps': vinfo.fps,
        'vinfo': vinfo._asdict()},
    'views_xywh': views_xywh,
    'segdata': segdata,
    'coms_2d': com2d,
    'dilate_segdata': segdata,
    'nclass': nclass,
    'nframe': nframe
    }
    # save  file
    outpkl  = osp.splitext(vfile)[0] + '.segpkl'
    pickle.dump(outdata, open(outpkl, 'wb'))
    print('saved to', outpkl)
    return outpkl
      

def parse_args(parser:argparse.ArgumentParser):
    args = parser.parse_args()
    video_path = args.video_path
    assert osp.exists(video_path), "video_path not exists"
    if osp.isfile(video_path):
        videos_path = [video_path]
    elif osp.isdir(video_path):
        videos_path = glob.glob(osp.join(video_path, "*.mp4"))
        videos_path = filter_vname(videos_path)
        assert len(videos_path) > 0, "no video found"
    else:
        raise ValueError("video_path is not a file or folder")

    print('total vfiles:', len(videos_path))
    num_gpus = min(torch.cuda.device_count()*2, len(videos_path))
    print("num_gpus:", num_gpus)

    return num_gpus, videos_path, args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video_path", type=str, default=None, help="path to video or folder"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--maxlen", type=int, default=None)

    num_gpus, videos_path, args = parse_args(parser)
    args_iterable_ = videos_path

    if True:
        mmap_cuda.workerpool_init(range(num_gpus), MyWorker, args.checkpoint, args.maxlen)
        mmap_cuda.workerpool_compute_map(args_iterable_)
    else:
        worker = MyWorker(args.checkpoint, args.maxlen)
        for args_ in args_iterable_:
            outdata = worker.compute(args_)
