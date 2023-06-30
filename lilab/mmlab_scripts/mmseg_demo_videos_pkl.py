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

from mmseg.apis import inference_segmentor, init_segmentor
from mmdet.core import encode_mask_results


ctx = mp.get_context('spawn')
queue_poolid = Manager().Queue()
workerpools = []  #manager.list()
nclass = 2   #white, black

class Worker:
    def __init__(self, config, checkpoint, ngpus, nfiles):
        super().__init__()
        self.id = queue_poolid.get()
        print(f'Worker #{self.id} init!')
        self.model = init_segmentor(config, checkpoint, device=f'cuda:{self.id}')

        print(f'Worker #{self.id} finished!')

    def compute(self, video, queue_poolid):
        out_pkl = osp.splitext(video)[0] + '.pkl'
        video_reader = mmcv.VideoReader(video)

        outputs = []
        for frame in tqdm.tqdm(video_reader, position=self.id):
            result = inference_segmentor(self.model, frame)
            mask = result[0]
            bboxes, segms = [], []
            for i in range(1, nclass+1):
                # test a single class
                mask_i = np.array(mask == i, dtype=np.uint8)
                if np.sum(mask_i) == 0:
                    bboxes.append([])
                    segms.append([])
                    continue
                contours, _ = cv2.findContours(mask_i, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # find the contours area
                areas = [cv2.contourArea(c) for c in contours]
                # keep the contours which area > 10
                contours = [contour for contour, area in zip(contours, areas) if area > 10]
                if contours:
                    mask_i_copy = np.zeros_like(mask_i, dtype=mask_i.dtype)
                    cv2.fillPoly(mask_i_copy, pts =contours, color=(1,1,1))
                    # find the bounding box of the mask_i_copy
                    x, y, w, h = cv2.boundingRect(mask_i_copy)
                    bboxes.append([np.array([x, y, x+w, y+h, 1])])
                    segms.append(encode_mask_results([[mask_i_copy]])[0])
                else:
                    bboxes.append([])
                    segms.append([])
            outputs.append([bboxes, segms])
        # save to pkl
        mmcv.dump(outputs, out_pkl)
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


def main(folder, config, checkpoint, num_gpus):
    videos = glob.glob(osp.join(folder, '*.mp4')) + glob.glob(
        osp.join(folder, '*.avi'))
    for poolid in range(num_gpus):
        queue_poolid.put(poolid)
    create_worker(config, checkpoint, num_gpus, len(videos))

    for poolid in range(num_gpus):
        queue_poolid.put(poolid)

    pool = ctx.Pool(processes=num_gpus)
    for video in videos:
        print('video:', video)
        pool.apply_async(compute, args=(video, queue_poolid, workerpools))

    pool.close()
    pool.join()
    print('End of pool.close()')


if __name__ == '__main__':
    args = parse_args()
    main(args.videos, args.config, args.checkpoint, args.num_gpus)
