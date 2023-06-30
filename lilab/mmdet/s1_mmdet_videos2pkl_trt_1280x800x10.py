# python -m lilab.mmdet.s1_mmdet_videos2pkl_trt  A.mp4 [CONFIG] [CHECKPOINT]
# python -m lilab.mmdet.s1_mmdet_videos2pkl_trt  A/B/C
import argparse
import mmcv
import numpy as np
import tqdm
import torch
import os.path as osp
import glob
from mmdet.apis import init_detector
from mmdet.core import encode_mask_results
from mmdet.datasets.pipelines import Compose
import lilab.cvutils.map_multiprocess_cuda as mmap_cuda
import ffmpegcv
from mmdet.apis import inference_detector
from mmdet2trt.apis import create_wrap_detector
from torchvision.transforms import functional as F
from torch.utils.data import IterableDataset, DataLoader
import itertools
from lilab.cameras_setup import get_view_xywh_1280x800x10 as get_view_xywh

video_path = [f for f in glob.glob('/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/test/*.mp4')
                if f[-4] not in '0123456789']

config = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/mask_rcnn_r101_fpn_2x_coco_bwrat.py'

def find_checkpoint(config):
    basedir = osp.dirname(config)
    basenakename = osp.splitext(osp.basename(config))[0]
    checkpoint = osp.join(basedir, 'work_dirs', basenakename, 'latest.trt')
    assert osp.isfile(checkpoint), 'checkpoint not found: {}'.format(checkpoint)
    return checkpoint

num_gpus = min([torch.cuda.device_count(), len(get_view_xywh())])

pos_views = get_view_xywh()


def prefetch_img_metas(cfg, ori_wh):
    w, h = ori_wh
    cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = {'img': np.zeros((h, w, 3), dtype=np.uint8)}
    data = test_pipeline(data)
    img_metas = data['img_metas'][0].data
    return img_metas


def process_img(frame_resize, img_metas):
    if isinstance(frame_resize, np.ndarray):
        assert frame_resize.shape == img_metas['pad_shape']
        frame_cuda = torch.from_numpy(frame_resize.copy()).cuda().float()
        frame_cuda = frame_cuda.permute(2, 0, 1)  # HWC to CHW
        frame_cuda = frame_cuda[None, :, :, :]  # NCHW
    else:
        frame_cuda = frame_resize.cuda().float()  # NHWC
        frame_cuda = frame_cuda.permute(0, 3, 1, 2)  # NCHW

    mean = torch.from_numpy(img_metas['img_norm_cfg']['mean']).cuda()
    std = torch.from_numpy(img_metas['img_norm_cfg']['std']).cuda()
    frame_cuda = F.normalize(frame_cuda, mean=mean, std=std, inplace=True)
    data = {'img': [frame_cuda], 'img_metas': [[img_metas]]}
    return data




class MyWorker(mmap_cuda.Worker):
# class MyWorker():
    def __init__(self, config, checkpoint):
        super().__init__()
        self.cuda = getattr(self, 'cuda', 0)
        self.id   = getattr(self, 'id', 0)
        self.config = config
        self.checkpoint = checkpoint
        print("well setup worker:", self.cuda)

    def compute(self, args):
        with torch.cuda.device(self.cuda), torch.no_grad():
            model = create_wrap_detector(self.checkpoint, self.config, 'cuda')
            video, iview = args
            crop_xywh = pos_views[iview]
            img_metas = prefetch_img_metas(model.cfg, crop_xywh[2:])
            resize_wh = img_metas['pad_shape'][1::-1]

            vid = ffmpegcv.VideoReaderNV(video, 
                                        crop_xywh=crop_xywh,
                                        resize = resize_wh,
                                        resize_keepratio = True,
                                        resize_keepratioalign = 'topleft',
                                        gpu = int(self.cuda),
                                        pix_fmt='rgb24')
            outputs = []
            out_pkl = video.replace('.mp4', f'_{iview}.pkl')
            for i, frame in enumerate(tqdm.tqdm(vid, position=self.id, desc=f'[{self.id}]')):
                data = process_img(frame, img_metas)
                result = model(return_loss=False, rescale=True, **data)
                if len(result)==2:
                    result = [result]
                result = [(bbox_results, encode_mask_results(mask_results))
                            for bbox_results, mask_results in result]
                resultf = filt_by_thr(result)
                outputs.extend(resultf)
            
        mmcv.dump(outputs, out_pkl)
        
        # from lilab.mmlab_scripts.detpkl_to_segpkl import convert
        # out_seg_pkl = convert(out_pkl)     #max p filter
        # from lilab.mmlab_scripts.segpkl_smooth import convert
        # out_seg_pkl = convert(out_seg_pkl)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, default=None, help='path to video or folder')
    parser.add_argument('--config', type=str, default=config)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    video_path = args.video_path
    assert osp.exists(video_path), 'video_path not exists'
    if osp.isfile(video_path):
        video_path = [video_path]
    elif osp.isdir(video_path):
        video_path = [f for f in glob.glob(osp.join(video_path, '*.mp4'))
                        if f[-4] not in '0123456789']
        assert len(video_path) > 0, 'no video found'
    else:
        raise ValueError('video_path is not a file or folder')
    if args.checkpoint is None:
        args.checkpoint = find_checkpoint(args.config)
    args_iterable = itertools.product(video_path, range(len(pos_views)))

    # init the workers pool
    mmap_cuda.workerpool_init(range(num_gpus), MyWorker, args.config, args.checkpoint)
    mmap_cuda.workerpool_compute_map(args_iterable)

    # worker = MyWorker(config, checkpoint)
    # worker.compute(args_iterable[0])
