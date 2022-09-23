# python -m lilab.mmpose.s1_rat_videos2pkl_trt E:/cxf/mmpose_rat/A.mp4
import argparse
import os.path as osp
import glob
import mmcv
import numpy as np
import tqdm
import torch
import pickle
import pycocotools.mask as maskUtils
from mmpose.apis import init_pose_model
from mmpose.datasets import DatasetInfo
import lilab.cvutils.map_multiprocess_cuda as mmap_cuda
import ffmpegcv
from torchvision.transforms import functional as F
from torch2trt import TRTModule
import itertools
import cv2
from lilab.cameras_setup import get_view_xywh_800x600x6 as get_view_xywh

config = '/home/liying_lab/chenxinfeng/DATA/mmpose/hrnet_w32_coco_udp_rattopdown_512x512.py'
pos_views = get_view_xywh()
num_gpus = min([torch.cuda.device_count(), len(get_view_xywh())])

checkpoint = None
if checkpoint is None:
    config = osp.abspath(config)
    basedir = osp.dirname(config)
    basenakename = osp.splitext(osp.basename(config))[0]
    checkpoint = osp.join(basedir, 'work_dirs', basenakename, 'latest.trt')
    assert osp.isfile(checkpoint), 'checkpoint not found: {}'.format(checkpoint)

def box2cs(box, image_size):
    """Encode bbox(x,y,w,h) into (center, scale) without padding.

    Returns:
        tuple: A tuple containing center and scale.
    """
    x, y, w, h = box[:4]

    aspect_ratio = 1. * image_size[0] / image_size[1]
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / 200.0, h * 1.0 / 200.0], dtype=np.float32)
    return center, scale


def prefetch_img_metas(cfg, ori_wh):
    """Pre-fetch the img_metas from config and original image size.

    Return:
        dict: img_metas.
    """
    w, h = ori_wh
    bbox = np.array([0, 0, w, h])
    center, scale = box2cs(bbox, cfg.data_cfg['image_size'])
    dataset_info = cfg.data['test'].get('dataset_info', None)
    dataset_info = DatasetInfo(dataset_info)
    assert dataset_info, 'Please set `dataset_info` in the config.'
    img_metas = {
        'img_or_path':
        None,
        'img':
        None,
        'image_file':
        '',
        'center':
        center,
        'scale':
        scale,
        'bbox_score':
        1,
        'bbox_id':
        0,
        'dataset':
        dataset_info.dataset_name,
        'joints_3d':
        np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
        'joints_3d_visible':
        np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
        'rotation':
        0,
        'ann_info': {
            'image_size': np.array(cfg.data_cfg['image_size']),
            'num_joints': cfg.data_cfg['num_joints'],
        },
        'flip_pairs':
        dataset_info.flip_pairs
    }
    for pipeline in cfg.test_pipeline[1:]:
        if pipeline['type'] == 'NormalizeTensor':
            img_metas['img_norm_cfg'] = {
                'mean': np.array(pipeline['mean']) * 255.0,
                'std': np.array(pipeline['std']) * 255.0
            }
            break
    else:
        raise Exception('NormalizeTensor is not found.')

    return img_metas


def process_img(frame_cuda, img_metas, device):
    """Process the image.

    Cast the image to device and do normalization.
    """
    # frame_cuda = torch.from_numpy(frame_resize).to(device).float()
    frame_cuda = frame_cuda.permute(2, 0, 1)  # HWC to CHW
    assert frame_cuda.shape[-1:-3:-1] == tuple(
        img_metas['ann_info']['image_size'])
    mean = torch.from_numpy(img_metas['img_norm_cfg']['mean']).to(device)
    std = torch.from_numpy(img_metas['img_norm_cfg']['std']).to(device)
    frame_cuda = F.normalize(frame_cuda, mean=mean, std=std, inplace=True)
    frame_cuda = frame_cuda[None, :, :, :]  # NCHW
    data = {'img': frame_cuda, 'img_metas': [img_metas]}
    return data


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
        self.return_heatmap = False
        self.output_layer_names = None
        self.pose_model = pose_model
        self.checkpoint = checkpoint
        print("well setup worker:", self.cuda)

    def compute(self, args):
        video, iview = args
        out_file = f'{video.replace(".mp4", "")}_{iview}.kptpkl'
        # if osp.exists(out_file): 
        #     print("skip:", osp.basename(out_file))
        #     return

        # pytorch set devide
        torch.cuda.set_device(self.cuda)
        trt_model = TRTModule()
        trt_model.load_state_dict(torch.load(self.checkpoint))

        model = self.pose_model
        cfg = model.cfg
        image_size = cfg.data_cfg.image_size

        crop_xywh = pos_views[iview]
        img_metas = prefetch_img_metas(model.cfg, crop_xywh[2:])
        vid = ffmpegcv.VideoReaderNV(video, 
                                    resize=image_size,
                                    resize_keepratio=True, 
                                    crop_xywh=crop_xywh,
                                    gpu = int(self.cuda),
                                    pix_fmt='rgb24')
        device = torch.device('cuda:{}'.format(self.cuda))
        segpkl  = osp.splitext(video)[0] + '.segpkl'
        segpkl_data = pickle.load(open(segpkl, 'rb'))
        segdata = segpkl_data['dilate_segdata'][iview]
        nclass = len(segdata[0])
        kpt_data = [[] for _ in range(nclass)]
        for i, (frame, segdata_now) in enumerate(zip(tqdm.tqdm(vid,position=int(self.id), 
                                    desc='worker[{}]'.format(self.id)), segdata)):
            frame_cuda = torch.from_numpy(frame).cuda().half()
            for iclass in range(nclass):
                mask = segdata_now[1][iclass]
                mask = maskUtils.decode(mask)[:,:,0]
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask, frame.shape[:2], cv2.INTER_NEAREST)
                maskcuda = torch.from_numpy(mask).cuda().half()
                frame_cuda_w_mask = frame_cuda * maskcuda[:, :, None]
                data = process_img(frame_cuda_w_mask, img_metas, device)
                heatmap = trt_model(data['img'])
                heatmap_np = heatmap.cpu().numpy()
                result = model.keypoint_head.decode([img_metas],
                                                    heatmap_np,
                                                    img_size=image_size)
                kpt_data[iclass].append(result['preds'][0])
        kpt_data = np.array(kpt_data).transpose([1,0,2,3]) # [times, nclass, njoints, xyp]
        out_dict = {f'{iview}': kpt_data, 
                     'info': segpkl_data['info'],
                     'crop_xywh': segpkl_data['views_xywh']}
        
        pickle.dump(out_dict, open(out_file, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, default=None, help='path to video or folder')
    parser.add_argument('--config', type=str, default=config)
    parser.add_argument('--checkpoint', type=str, default=checkpoint)
    arg = parser.parse_args()

    video_path = arg.video_path
    assert osp.exists(video_path), 'video_path not exists'
    if osp.isfile(video_path):
        video_path = [video_path]
    elif osp.isdir(video_path):
        video_path = glob.glob(osp.join(video_path, '*.mp4'))
        assert len(video_path) > 0, 'no video found'
    else:
        raise ValueError('video_path is not a file or folder')

    config, checkpoint = arg.config, arg.checkpoint
    args_iterable = list(itertools.product(video_path, range(len(pos_views))))
    # init the workers pool
    # mmap_cuda.workerpool_init(range(num_gpus), MyWorker, config, checkpoint)
    # mmap_cuda.workerpool_compute_map(args_iterable)

    worker = MyWorker(config, checkpoint)
    for args in args_iterable:
        worker.compute(args)
    