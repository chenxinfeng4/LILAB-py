# python -m lilab.multiview_scripts_new.s1_ballvideo2matpkl  A.mp4
# %%
import argparse
import os
import re
import os.path as osp
import glob
import mmcv
import numpy as np
import tqdm
import torch
import pickle
from mmpose.apis import init_pose_model
from mmpose.apis.inference import Compose, collate, _box2cs
from mmpose.datasets import DatasetInfo
import lilab.cvutils.map_multiprocess_cuda as mmap_cuda
import ffmpegcv
from ffmpegcv.video_info import get_info
from torch2trt import TRTModule
import itertools
# from lilab.cameras_setup import get_view_xywh_1280x800x10 as get_view_xywh
from lilab.cameras_setup import get_view_xywh_800x600x6 as get_view_xywh

config = '/home/liying_lab/chenxinfeng/DATA/mmpose/hrnet_w32_coco_ball_512x512.py'
num_gpus = min([torch.cuda.device_count(), len(get_view_xywh())])

pos_views = get_view_xywh()


checkpoint = None
if checkpoint is None:
    config = osp.abspath(config)
    basedir = osp.dirname(config)
    basenakename = osp.splitext(osp.basename(config))[0]
    checkpoint = osp.join(basedir, 'work_dirs', basenakename, 'latest.trt')
    assert osp.isfile(checkpoint), 'checkpoint not found: {}'.format(checkpoint)


def convert2matpkl(vfile):
    vinfo = get_info(vfile)

    pkl_files = glob.glob(osp.splitext(vfile)[0] + '_*.kptpkl')
    p = re.compile('.*(\d+)\.kptpkl$')
    views = [int(p.findall(pkl_file)[0]) for pkl_file in pkl_files]
    assert len(views) == max(views)+1 and min(views) == 0

    outdata = { 'info': {
                    'vfile': vfile, 
                    'nview': len(views), 
                    'fps': vinfo.fps,
                    'vinfo': vinfo._asdict()},
                'views_xywh': pos_views,
                'keypoints': {} }

    keypoints = [[] for _ in range(len(views))]

    for view, pkl_file in zip(views, pkl_files):
        data = pickle.load(open(pkl_file, 'rb'))
        assert isinstance(data, dict) and str(view) in data
        keypoints[view] = data[str(view)]

    keypoints = np.array(keypoints)
    outdata['keypoints'] = keypoints

    # %% save to mat file
    outpkl  = osp.splitext(vfile)[0] + '.matpkl'
    pickle.dump(outdata, open(outpkl, 'wb'))
    print('saved to', outpkl)

    # %% remove the pkl files
    for pkl_file in pkl_files:
        os.remove(pkl_file)
    return outpkl

class DataSet():
    def __init__(self, vid, device, cfg, dataset_info):
        self.vid = vid
        w, h = vid.crop_width, vid.crop_height

        imagesize = [vid.width, vid.height]
        assert cfg.data_cfg['image_size'] == imagesize

        person_results = [{'bbox': np.array([0, 0, w, h])}]
        bboxes_xywh = np.array([box['bbox'] for box in person_results])

        test_pipeline = cfg.test_pipeline[2:] # the affine transform is not needed
        test_pipeline = Compose(test_pipeline)
        for t in test_pipeline.transforms:
            if t.__class__.__name__ == 'ToTensor':
                t.device = device

        dataset_name = dataset_info.dataset_name
        flip_pairs = dataset_info.flip_pairs
        bbox = bboxes_xywh[0]
        center, scale = _box2cs(cfg, bbox)
        scale /= 1.25           # I don't wanna expand bbox area
        data_template = {
            'img_or_path': None,
            'img': None,
            'image_file': '',
            'center': center,
            'scale':scale,
            'bbox_score': bbox[4] if len(bbox) == 5 else 1,
            'bbox_id': 0, 
            'dataset': dataset_name,
            'joints_3d':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'joints_3d_visible':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'rotation': 0,
            'ann_info': {
                'image_size': np.array(cfg.data_cfg['image_size']),
                'num_joints': cfg.data_cfg['num_joints'],
                'flip_pairs': flip_pairs
            }
        }
        self.data_template = data_template
        self.test_pipeline = test_pipeline

    def __iter__(self):
        for img in self.vid:
            # the img is HxWxC, and color channel is RGB
            data_contain = self.data_template.copy()
            data_contain['img'] = img
            data_ready = self.test_pipeline(data_contain)
            yield data_ready

    def __len__(self):
        return len(self.vid)


class MyWorker(mmap_cuda.Worker):
# class MyWorker:
    def __init__(self, config, checkpoint):
        super().__init__()
        self.id = getattr(self, 'id', 0)
        self.cuda = getattr(self, 'cuda', 0)
        pose_model = init_pose_model(
                        config, checkpoint=None, device='cpu')
        self.dataset = pose_model.cfg.data['test']['type']
        dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
        assert dataset_info, 'Please set `dataset_info` in the config.'
        self.dataset_info = DatasetInfo(dataset_info)
        self.return_heatmap = False
        self.output_layer_names = None
        self.pose_model = pose_model
        self.checkpoint = checkpoint
        print("well setup worker:", self.cuda)

    def compute(self, args):
        with torch.cuda.device(self.cuda):
            trt_model = TRTModule()
            trt_model.load_state_dict(torch.load(self.checkpoint))

        model = self.pose_model
        dataset_info = self.dataset_info
        cfg = model.cfg
        image_size = cfg.data_cfg.image_size
        video, iview = args
        crop_xywh = pos_views[iview]
        vid = ffmpegcv.VideoReaderNV(video, 
                                    resize=image_size,
                                    resize_keepratio=True, 
                                    crop_xywh=crop_xywh,
                                    gpu = int(self.cuda),
                                    pix_fmt='rgb24')
        device = torch.device('cuda:{}'.format(self.cuda))
        dataset = DataSet(vid, device, cfg, dataset_info)
        kpt_data = []
        for data in tqdm.tqdm(dataset,position=int(self.id), 
                              desc='worker[{}]'.format(self.id)):
            batch_data = collate([data], 
                                 samples_per_gpu=1)
            batch_data['img'] = batch_data['img'].to(device, dtype=torch.float16)  #380fps
            batch_data['img_metas'] = [
                img_metas for img_metas in batch_data['img_metas'].data[0]
            ]
            with torch.cuda.device(self.cuda):
                heatmap = trt_model(batch_data['img'])
            heatmap_np = heatmap.cpu().numpy()
            result = model.keypoint_head.decode(batch_data['img_metas'],
                                                heatmap_np,
                                                img_size=image_size)
            kpt_data.append(result['preds'])
        kpt_data = np.concatenate(kpt_data, axis=0)
        pickle.dump({f'{iview}': kpt_data}, 
                    open(f'{video.replace(".mp4", "")}_{iview}.kptpkl','wb'))


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, default=None, help='path to video or folder')
    parser.add_argument('--config', type=str, default=config)
    parser.add_argument('--checkpoint', type=str, default=checkpoint)
    arg = parser.parse_args()

    video_path, config, checkpoint = arg.video_path, arg.config, arg.checkpoint
    assert osp.exists(video_path), 'video_path not exists'
    if osp.isfile(video_path):
        video_path = [video_path]
    elif osp.isdir(video_path):
        video_path = glob.glob(osp.join(video_path, '*.mp4'))
        assert len(video_path) > 0, 'no video found'
    else:
        raise ValueError('video_path is not a file or folder')

    args_iterable = itertools.product(video_path, range(len(pos_views)))
    # init the workers pool
    mmap_cuda.workerpool_init(range(num_gpus), MyWorker, config, checkpoint)
    mmap_cuda.workerpool_compute_map(args_iterable)

    # worker = MyWorker(config, checkpoint)
    # for args in args_iterable:
    #     worker.compute(args)

    # post_process pkl files to matpkl
    for video in video_path:
        convert2matpkl(video)
        print('python -m lilab.multiview_scripts_new.s2_matpkl2ballpkl',
              video.replace('.mp4', '.matpkl'),
              '--time 1 2 3 4 5')
