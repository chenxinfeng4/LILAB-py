# python -m lilab.mmdet_dev.s1_mmdet_videos2pkl_trt  A.mp4 --pannels 9 [CONFIG] [CHECKPOINT]
# python -m lilab.mmdet_dev.s1_mmdet_videos2pkl_trt  A/B/C
import argparse
import mmcv
import numpy as np
import tqdm
import torch
import os
import os.path as osp
import glob
from mmdet.core import encode_mask_results
from mmdet.datasets.pipelines import Compose
import lilab.cvutils.map_multiprocess_cuda as mmap_cuda
import ffmpegcv
from mmdet2trt.apis import create_wrap_detector
from torchvision.transforms import functional as F
import itertools
from lilab.cameras_setup import get_view_xywh_wrapper
import lilab.cvutils.map_multiprocess as mmap
from lilab.mmdet_dev.s2_detpkl_to_segpkl import convert as convert_detpkl_to_segpkl
from lilab.mmdet_dev.s2_segpkl_merge import convert as convert_segpkl_to_one
# from lilab.mmpose_dev.a2_convert_mmpose2engine import findcheckpoint_trt

video_path = [f for f in glob.glob('/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/test/*.mp4')
                if f[-4] not in '0123456789']

# config = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/mask_rcnn_r101_fpn_2x_coco_bwrat_800x600.py'
config = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/mask_rcnn_r101_fpn_2x_coco_bwrat_816x512_cam9.py'
# config = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/mask_rcnn_r101_fpn_2x_coco_bwrat_816x512_cam9_oldrat.py'

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

def findcheckpoint_trt(config, trtnake='latest.engine'):
    """Find the latest checkpoint of the model."""
    basedir = osp.dirname(config)
    basenakename = osp.splitext(osp.basename(config))[0]
    checkpoint = osp.join(basedir, 'work_dirs', basenakename, trtnake)
    assert osp.isfile(checkpoint), 'checkpoint not found: {}'.format(checkpoint)
    return checkpoint

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
        video, (iview, crop_xywh) = args
        out_pkl = osp.splitext(video)[0] + f'_{iview}.pkl'
        if os.path.exists(out_pkl): 
            print("Skipping:", osp.basename(out_pkl))
            return out_pkl
            
        with torch.cuda.device(self.cuda), torch.no_grad():
            model = create_wrap_detector(self.checkpoint, self.config, 'cuda')
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
            for frame in tqdm.tqdm(vid, position=self.id, desc=f'[{self.id}]'):
                data = process_img(frame, img_metas)
                result = model(return_loss=False, rescale=True, **data)
                
                if len(result)==2:
                    result = [result]
                result = [(bbox_results, encode_mask_results(mask_results))
                            for bbox_results, mask_results in result]
                # continue
                resultf = filt_by_thr(result)
                outputs.extend(resultf)

        mmcv.dump(outputs, out_pkl)
        return out_pkl
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
    parser.add_argument('--pannels', type=int, default=4, help='crop views')
    parser.add_argument('--config', type=str, default=config)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    views_xywh = get_view_xywh_wrapper(args.pannels)
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
        args.checkpoint = findcheckpoint_trt(args.config, 'latest.trt')
    args_iterable = list(itertools.product(video_path, enumerate(views_xywh)))

    print(torch.cuda.device_count(), len(args_iterable))
    num_gpus = min((torch.cuda.device_count(), len(args_iterable)))
    print('num_gpus:', num_gpus)
    # init the workers pool
    mmap_cuda.workerpool_init(range(num_gpus), MyWorker, args.config, args.checkpoint)
    detpkls = mmap_cuda.workerpool_compute_map(args_iterable)

    # worker = MyWorker(args.config, args.checkpoint)
    # detpkls = [worker.compute(args) for args in args_iterable]

    # detpkl to segpkl
    # segpkls = mmap.map(convert_detpkl_to_segpkl, detpkls)

    # # merge segpkl to one
    # mmap.map(convert_segpkl_to_one, video_path)
    # for segpkl in segpkls:
    #     os.remove(segpkl)

    # # print done
    # print('done')
    # print('python lilab.mmdet_dev.s2_segpkl_dilate  xxx')
