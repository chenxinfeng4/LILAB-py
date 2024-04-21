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
from mmdet.core import encode_mask_results
from multiprocessing import Process, Queue
import itertools
from lilab.mmdet_dev.filter_vname import filter_vname
from torch2trt import TRTModule
from lilab.yolo_seg.utilities import (singleton_gpu, refine_mask2, refine_mask3)
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
    

def create_segpkl(q:Queue, q2:Queue):

    result_all = []
    com2d_all = []
    iframes = []
    outdata = dict()
    while True:
        iframe, result = q.get()
        if iframe is None:
            break
        # continue
        result = s1_filt_by_thr(result)# video2detpkl part
        result = s2_det2seg_part(result)# detpkl2segpkl part
        result = s3_dilate_cv(result)  # segpkl dilate part
        coms_real_2d = s4_com2d(result)  # segpkl com2d part
    
        result_encode = [(bbox_results, encode_mask_results(mask_results))
        for bbox_results, mask_results in result]

        result_all.append(result_encode[0])
        com2d_all.append(coms_real_2d)
        iframes.append(iframe)

    outdata['segdata'] = result_all
    outdata['coms_2d'] = com2d_all
    outdata['iframes'] = iframes

    q2.put(outdata)



# class MyWorker(mmap_cuda.Worker):
class MyWorker():
    def __init__(self, checkpoint, maxlen):
        super().__init__()
        self.cuda = getattr(self, "cuda", 0)
        self.id = getattr(self, "id", 0)
        self.checkpoint = checkpoint
        self.maxlen = maxlen
        print("well setup worker:", self.cuda)
        self.cworker_per_gpu = 4

    def compute(self, args):
        video = args
        
        model, input_shape = load_trtmodel(self.checkpoint, self.cuda)
        resize_WH = input_shape[::-1]
        vid = ffmpegcv.VideoCaptureNV(video, resize=resize_WH, resize_keepratio=False,
                                      gpu=int(self.cuda),pix_fmt="gray")

        maxlen = min([len(vid), self.maxlen]) if self.maxlen else len(vid)
        # context = mp.get_context()
        # q = context.Queue(maxsize=100)
        # q2 = context.Queue(maxsize=10)
        # c_process = [context.Process(target=create_segpkl, args=(q,q2,img_metas,model.CLASSES)) for _ in range(self.cworker_per_gpu)]
        # _ = [process.start() for process in c_process]

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
    num_gpus = min(torch.cuda.device_count(), len(videos_path))
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
    # mmap_cuda.workerpool_init(range(num_gpus), MyWorker, args.checkpoint, args.maxlen)
    
    # detpkls = mmap_cuda.workerpool_compute_map(args_iterable)
    worker = MyWorker(args.checkpoint, args.maxlen)
    for args_ in videos_path:
       outdata = worker.compute(args_)
