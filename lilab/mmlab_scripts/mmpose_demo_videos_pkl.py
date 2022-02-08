# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import multiprocessing as mp
import os
import os.path as osp
import warnings
from multiprocessing import Manager, Pool, Queue

import mmcv
import numpy as np
import torch.multiprocessing as mp
import tqdm

from mmpose.apis import inference_bottom_up_pose_model, init_pose_model
from mmpose.datasets import DatasetInfo

ctx = mp.get_context('spawn')
queue_poolid = Manager().Queue()
workerpools = []  #manager.list()


class Worker:

    def __init__(self, config, checkpoint, ngpus, nfiles):
        super().__init__()
        self.id = queue_poolid.get()
        print(f'Worker #{self.id} init!')
        pose_model = init_pose_model(
            config, checkpoint, device=f'cuda:{self.id}')
        dataset = pose_model.cfg.data['test']['type']
        dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
        if dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
            assert (dataset == 'BottomUpCocoDataset')
        else:
            dataset_info = DatasetInfo(dataset_info)

        num_joint = len(dataset_info.keypoint_info)

        # optional
        return_heatmap = False

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None
        pose_nms_thr = 0.9

        # self
        self.pose_model = pose_model
        self.num_joint = num_joint
        self.return_heatmap = return_heatmap
        self.output_layer_names = output_layer_names
        self.pose_nms_thr = pose_nms_thr
        self.ngpus = ngpus
        self.nfiles = nfiles
        self.dataset = dataset
        self.dataset_info = dataset_info
        print(f'Worker #{self.id} finished!')

    def compute(self, video, queue_poolid):
        out_pkl = osp.splitext(video)[0] + '.pkl'
        video_reader = mmcv.VideoReader(video)

        outputs = []
        for img in tqdm.tqdm(video_reader, position=self.id):
            pose_results, returned_outputs = inference_bottom_up_pose_model(
                self.pose_model,
                img,
                dataset=self.dataset,
                dataset_info=self.dataset_info,
                pose_nms_thr=self.pose_nms_thr,
                return_heatmap=self.return_heatmap,
                outputs=self.output_layer_names)

            if len(pose_results) == 0:
                keypoints = np.zeros((self.num_joint, 3))
            elif len(pose_results) == 1:
                keypoints = pose_results[0]['keypoints']
            else:
                #sort the pose_results by the max score
                pose_results = sorted(
                    pose_results, key=lambda x: x['score'], reverse=True)
                keypoints = pose_results[0]['keypoints']
            outputs.append(keypoints)
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
