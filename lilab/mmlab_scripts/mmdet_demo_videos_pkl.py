# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import multiprocessing as mp
import os
import os.path as osp
import warnings
from multiprocessing import Manager, Pool, Queue

import cv2
import mmcv
import numpy as np
import torch.multiprocessing as mp
import tqdm

from mmdet.apis import inference_detector as inference_model, init_detector as init_model
from mmdet.core import encode_mask_results
from lilab.mmlab_scripts.detpkl_to_segpkl import convert as detpkl_to_segpkl

ctx = mp.get_context('spawn')
queue_poolid = Manager().Queue()
workerpools = []  #manager.list()


class Worker:
    def __init__(self, config, checkpoint, ngpus, nfiles):
        super().__init__()
        self.id = queue_poolid.get()
        print(f'Worker #{self.id} init!')
        self.model = init_model(config, checkpoint, device=f'cuda:{self.id}')

        print(f'Worker #{self.id} finished!')

        # self
        self.ngpus = ngpus
        self.nfiles = nfiles

    def compute(self, video, queue_poolid):
        out_pkl = osp.splitext(video)[0] + '.pkl'
        video_reader = mmcv.VideoReader(video)

        outputs = []
        for frame in tqdm.tqdm(video_reader, position=self.id):
            result = inference_model(self.model, frame)
            if len(result)==2:
                result = [result]
            result = [(bbox_results, encode_mask_results(mask_results))
                        for bbox_results, mask_results in result]
            resultf = filt_by_thr(result)
            outputs.extend(resultf)
        # save to pkl
        mmcv.dump(outputs, out_pkl) #save to det_pkl
        detpkl_to_segpkl(out_pkl)   #save to seg_pkl
        queue_poolid.put(self.id)


def create_worker(config, checkpoint, ngpus, nfiles):
    for i in range(ngpus):
        worker = Worker(config, checkpoint, ngpus, nfiles)
        workerpools.append(worker)


def compute(video, queue_poolid, workerpools):
    id = queue_poolid.get()
    workerpools[id].compute(video, queue_poolid)


def parse_args():
    parser = argparse.ArgumentParser(description='MMSegementation video demo')
    parser.add_argument('videos', help='Video folder')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--num_gpus', default=4, type=int, help='Device used for inference')
    args = parser.parse_args()
    return args


# mask post processing
def filt_by_thr(result, thr=0.5):
    result_out = []
    for a_frame_result in result:
        bbox_results, mask_results = a_frame_result
        a_frame_out = [[],[]]
        for a_class_bbox, a_class_mask in zip(bbox_results, mask_results):
            p_vals = a_class_bbox[:,-1]
            valid  = p_vals > thr
            a_class_bbox = a_class_bbox[valid]
            a_class_mask = [mask for mask,v in zip(a_class_mask,valid) if v]
            a_frame_out[0].append(a_class_bbox)
            a_frame_out[1].append(a_class_mask)

        result_out.append(a_frame_out)
    return result_out



def main(folder, config, checkpoint, num_gpus):
    videos = glob.glob(osp.join(folder, '*.mp4')) + glob.glob(
        osp.join(folder, '*.avi'))
    for poolid in range(num_gpus):
        queue_poolid.put(poolid)
    create_worker(config, checkpoint, num_gpus, len(videos))

    for poolid in range(num_gpus):
        queue_poolid.put(poolid)

    pool = ctx.Pool(processes=num_gpus)
    # pool = Pool(processes=num_gpus)
    for video in videos:
        print('video:', video)
        pool.apply_async(compute, args=(video, queue_poolid, workerpools))

    pool.close()
    pool.join()
    print('End of pool.close()')


if __name__ == '__main__':
    args = parse_args()
    main(args.videos, args.config, args.checkpoint, args.num_gpus)
