# python -m lilab.mmlab_scripts.mmpose_videos_to_pkl videos/ config checkpoint
import argparse
import glob
import os.path as osp
import mmcv
import numpy as np
import tqdm
import warnings
from mmpose.apis import inference_bottom_up_pose_model, init_pose_model
from mmpose.datasets import DatasetInfo


import lilab.cvutils.map_multiprocess_cuda as mmp_cuda

class MyWorker(mmp_cuda.Worker):
    def __init__(self, config, checkpoint):
        super().__init__()
        print(f'Worker #{self.id} init!')
        pose_model = init_pose_model(
                        config, checkpoint, device=f'cuda:{self.cuda}')
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
        self.dataset = dataset
        self.dataset_info = dataset_info
        print(f'Worker #{self.id} finished!')


    def compute(self, video):
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



def main(folder, config, checkpoint, num_gpus):
    videos = glob.glob(osp.join(folder, '*.mp4')) + glob.glob(
                        osp.join(folder, '*.avi'))
    mmp_cuda.workerpool_init(range(num_gpus), MyWorker, config, checkpoint)
    mmp_cuda.workerpool_compute_map(videos)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMSegementation video demo')
    parser.add_argument('videos', help='Video folder')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--num_gpus', default=4, type=int, help='Device used for inference')
    args = parser.parse_args()
    main(args.videos, args.config, args.checkpoint, args.num_gpus)
    