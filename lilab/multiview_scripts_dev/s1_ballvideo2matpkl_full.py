# python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full  A.mp4
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
from mmpose.datasets import DatasetInfo
import lilab.cvutils.map_multiprocess_cuda as mmap_cuda
import ffmpegcv
from ffmpegcv.video_info import get_info
from torch2trt import TRTModule
import itertools
from lilab.mmpose_dev.a2_convert_mmpose2engine import findcheckpoint_trt
from lilab.cameras_setup import get_view_xywh_wrapper
from lilab.multiview_scripts_dev.comm_functions import (
      box2cs, get_max_preds, get_max_preds_gpu, transform_preds)


config_dict = {6:'/home/liying_lab/chenxinfeng/DATA/mmpose/hrnet_w32_coco_ball_512x320.py',
          10:'/home/liying_lab/chenxinfeng/DATA/mmpose/res50_coco_ball_512x320.py',
          4: '/home/liying_lab/chenxinfeng/DATA/mmpose/res50_coco_ball_512x320_ZJF.py'}

num_gpus = min([torch.cuda.device_count(), 4])

pos_views = []


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
    def __init__(self, vid, cfg):
        self.vid = vid
        w, h = vid.crop_width, vid.crop_height

        bbox = np.array([0, 0, w, h])
        center, scale = box2cs(bbox, cfg.data_cfg['image_size'])
        data_template = {
            'img_metas':{
                'center': center,
                'scale': scale},
            'img':None}
        self.center, self.scale = center, scale
        self.data_template = data_template

    def __iter__(self):
        for img in self.vid:
            # the img change to CxHxW, and color channel is RGB
            img_CHW = torch.from_numpy(img.copy()).permute(2,0,1)
            yield img_CHW

    def __len__(self):
        return len(self.vid)


# class MyWorker(mmap_cuda.Worker):
class MyWorker:
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
        dataset = DataSet(vid, cfg)
        center, scale = dataset.center, dataset.scale
        kpt_data = []
        for img_CHW in tqdm.tqdm(dataset,position=int(self.id), 
                              desc='worker[{}]'.format(self.id)):
            batch_img = torch.unsqueeze(img_CHW, dim=0) #BxCxHxW
            batch_img = batch_img.to(device).half() #380fps
            with torch.cuda.device(self.cuda):
                heatmap = trt_model(batch_img)
                N, K, H, W = heatmap.shape

            # heatmap_np = heatmap.cpu().numpy()
            # preds, maxvals = get_max_preds(heatmap_np)
            preds, maxvals = get_max_preds_gpu(heatmap)
            preds = transform_preds(
                preds, center, scale, [W, H], use_udp=False)
            all_preds = np.concatenate((preds, maxvals), axis=-1)
            kpt_data.append(all_preds)
            
        kpt_data = np.concatenate(kpt_data, axis=0)
        pickle.dump({f'{iview}': kpt_data}, 
                    open(f'{video.replace(".mp4", "")}_{iview}.kptpkl','wb'))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, help='path to video or folder')
    parser.add_argument('--pannels', type=int, default=4, help='crop views')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    arg = parser.parse_args()

    pos_views[:] = get_view_xywh_wrapper(arg.pannels)
    video_path, config, checkpoint = arg.video_path, arg.config, arg.checkpoint
    if config is None:
        config = config_dict[arg.pannels]
    if checkpoint is None:
        checkpoint = findcheckpoint_trt(config, trtnake='latest.full.trt')
    print("config:", config)
    print("checkpoint:", checkpoint)

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
    # mmap_cuda.workerpool_init(range(num_gpus), MyWorker, config, checkpoint)
    # mmap_cuda.workerpool_compute_map(args_iterable)

    worker = MyWorker(config, checkpoint)
    for args in args_iterable:
        worker.compute(args)

    # post_process pkl files to matpkl
    for video in video_path:
        convert2matpkl(video)
        print('python -m lilab.multiview_scripts_new.s2_matpkl2ballpkl',
              video.replace('.mp4', '.matpkl'),
              '--time 1 2 3 4 5')
        print('python -m lilab.multiview_scripts_new.s5_show_calibpkl2video',
               video.replace('.mp4', '.matpkl'))
    