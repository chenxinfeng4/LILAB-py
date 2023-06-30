# python -m lilab.mmdet_dev.s1_mmdet_videos2pkl_trt  A.mp4  [CONFIG] [CHECKPOINT]
# python -m lilab.mmdet_single.s1_mmdet_videos2pkl_single  A/B/C
import argparse
import glob
import os
import os.path as osp

import ffmpegcv
import numpy as np
import torch
import tqdm
from mmdet2trt.apis import create_wrap_detector
from torchvision.transforms import functional as F

from mmdet.apis import init_detector
import lilab.cvutils.map_multiprocess as mmap
import lilab.cvutils.map_multiprocess_cuda as mmap_cuda
from lilab.cameras_setup import get_view_xywh_wrapper
from lilab.mmdet_dev.s2_detpkl_to_segpkl import convert as convert_detpkl_to_segpkl
from lilab.mmdet_dev.s2_segpkl_merge import convert as convert_segpkl_to_one
from mmdet.core import encode_mask_results
from mmdet.datasets.pipelines import Compose
from multiprocessing import Process, Queue
import multiprocess as mp

# from lilab.mmpose_dev.a2_convert_mmpose2engine import findcheckpoint_trt

video_path = [
    f
    for f in glob.glob(
        "/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/test/*.mp4"
    )
    if f[-4] not in "0123456789"
]

# config = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/mask_rcnn_r101_fpn_2x_coco_bwrat_800x600.py'
config = "/home/liying_lab/chenxinfeng/DATA/CBNetV2/mask_rcnn_r101_fpn_2x_coco_onemice_816x512.py"
# config = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/mask_rcnn_r101_fpn_2x_coco_bwrat_816x512_cam9_oldrat.py'

iview, crop_xywh = 0, [513, 75, 790, 700]
def prefetch_img_metas(cfg, ori_wh):
    w, h = ori_wh
    cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = {"img": np.zeros((h, w, 3), dtype=np.uint8)}
    data = test_pipeline(data)
    img_metas = data["img_metas"][0].data
    return img_metas


def process_img(frame_resize, img_metas):
    if isinstance(frame_resize, np.ndarray):
        assert frame_resize.shape == img_metas["pad_shape"]
        frame_cuda = torch.from_numpy(frame_resize.copy()).cuda().float()
        frame_cuda = frame_cuda.permute(2, 0, 1)  # HWC to CHW
        frame_cuda = frame_cuda[None, :, :, :]  # NCHW
    else:
        frame_cuda = frame_resize.cuda().float()  # NHWC
        frame_cuda = frame_cuda.permute(0, 3, 1, 2)  # NCHW

    mean = torch.from_numpy(img_metas["img_norm_cfg"]["mean"]).cuda()
    std = torch.from_numpy(img_metas["img_norm_cfg"]["std"]).cuda()
    frame_cuda = F.normalize(frame_cuda, mean=mean, std=std, inplace=True)
    data = {"img": [frame_cuda], "img_metas": [[img_metas]]}
    return data


def findcheckpoint_trt(config, trtnake="latest.engine"):
    """Find the latest checkpoint of the model."""
    basedir = osp.dirname(config)
    basenakename = osp.splitext(osp.basename(config))[0]
    checkpoint = osp.join(basedir, "work_dirs", basenakename, trtnake)
    assert osp.isfile(checkpoint), "checkpoint not found: {}".format(checkpoint)
    return checkpoint


def create_video(q:Queue, vfile_out:str, fps, cuda):
    vid_out = ffmpegcv.VideoWriterNV(
        vfile_out,
        codec='h264',
        fps=fps,
        gpu=int(cuda),
    )
    #the first frame is background
    frame_origin, mask = q.get()
    frame_canvas = np.right_shift(frame_origin, 1).astype('float32') + 0.5 *254
    frame_canvas[mask] = np.median(frame_canvas[~mask])
    frame_canvas = frame_canvas.astype('uint8')
    vid_out.write(frame_canvas)

    while True:
        frame_origin, mask = q.get()
        if frame_origin is None:
            break

        # frame_canvas = frame_origin.astype('float32')*0.5 + 0.5 *254
        frame_canvas = np.right_shift(frame_origin, 1).astype('float32') + 0.5 *254
        frame_canvas[mask] *= 0.2

        frame_canvas = frame_canvas.astype('uint8')
        vid_out.write(frame_canvas)

    vid_out.release()


# class MyWorker(mmap_cuda.Worker):
class MyWorker():
    def __init__(self, config, checkpoint, maxlen):
        super().__init__()
        self.cuda = getattr(self, "cuda", 0)
        self.id = getattr(self, "id", 0)
        self.config = config
        self.checkpoint = checkpoint
        self.maxlen = maxlen
        print("well setup worker:", self.cuda)

    def compute(self, args):
        video = args
        
        out_pkl = osp.splitext(video)[0] + f"_{iview}.pkl"
        if os.path.exists(out_pkl):
            print("Skipping:", osp.basename(out_pkl))
            return out_pkl
        
        with torch.cuda.device(self.cuda), torch.no_grad():
            model = create_wrap_detector(self.checkpoint, self.config, "cuda")
            # model = init_detector(self.config, self.checkpoint, 'cuda')
            img_metas = prefetch_img_metas(model.cfg, crop_xywh[2:])
            resize_wh = img_metas["pad_shape"][1::-1]
            img_metas['ori_shape'] = img_metas['img_shape'] = img_metas['pad_shape']
            img_metas['scale_factor'][:] = 1

        vid = ffmpegcv.VideoCaptureNV(
            video,
            crop_xywh=crop_xywh,
            resize=resize_wh,
            gpu=int(self.cuda),
            pix_fmt="rgb24",
        )
        
        vfile_out = video.replace('.mp4', '_mask.mp4')
        
        # context = mp.get_context('spawn')
        context = mp.get_context('fork')
        q = context.Queue(maxsize=30)
        process = context.Process(target=create_video, args=(q, vfile_out, vid.fps, int(self.cuda)))
        process.start()

        with torch.cuda.device(self.cuda), torch.no_grad(), vid:
            # model = create_wrap_detector(self.checkpoint, self.config, "cuda")
            # model = init_detector(self.config, self.checkpoint, 'cuda')
            # img_metas = prefetch_img_metas(model.cfg, crop_xywh[2:])
            # resize_wh = img_metas["pad_shape"][1::-1]

            maxlen = min([len(vid), self.maxlen]) if self.maxlen else len(vid)

            for i, frame in zip(tqdm.trange(maxlen, position=self.id, desc=f"[{self.id}]"), vid):
                if i>1000:
                    break
                data = process_img(frame, img_metas)
                result = model(return_loss=False, rescale=True, **data)

                if len(result) == 2:
                    result = [result]

                merge_mask = self.merge_masks(result, imghw=resize_wh[::-1])
                q.put((frame, merge_mask))
        q.put((None, None))
        process.join()


    def merge_masks(self, result, imghw, thr=0.7):
        defaultMask = getattr(self, 'defaultMask', np.zeros(imghw, dtype='bool'))
        masks = np.array(result[0][1][0])
        if not len(masks):
            return defaultMask
        pvals = result[0][0][0][:,-1]
        if pvals.max() > thr:
            self.defaultMask = masks[0]
            return masks[0]
        else:
            return defaultMask
        
    # masks_valid = masks[pvals > thr]
    # if masks_valid:
    #     masks_bool = np.any(masks_valid, axis=0)
    # else:
    #     masks_bool = np.zeros(imghw, dtype='bool')
    # return masks_bool



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video_path", type=str, default=None, help="path to video or folder"
    )
    parser.add_argument("--config", type=str, default=config)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--maxlen", type=int, default=None)

    args = parser.parse_args()

    video_path = args.video_path
    assert osp.exists(video_path), "video_path not exists"
    if osp.isfile(video_path):
        videos_path = [video_path]
    elif osp.isdir(video_path):
        videos_path = [
            f
            for f in glob.glob(osp.join(video_path, "*.mp4"))
            if  'mask.mp4' not in f
        ]
        assert len(videos_path) > 0, "no video found"
    else:
        raise ValueError("video_path is not a file or folder")
    if args.checkpoint is None:
        args.checkpoint = findcheckpoint_trt(args.config, "latest.trt")
        # args.checkpoint = findcheckpoint_trt(args.config, "latest.pth")

    args_iterable = videos_path

    print('total vfiles:', len(args_iterable))
    num_gpus = min((torch.cuda.device_count(), len(args_iterable)))
    print("num_gpus:", num_gpus)
    # init the workers pool
    
    # mmap_cuda.workerpool_init(range(num_gpus), MyWorker, args.config, args.checkpoint, args.maxlen)
    # detpkls = mmap_cuda.workerpool_compute_map(args_iterable)

    worker = MyWorker(args.config, args.checkpoint, args.maxlen)
    for args in args_iterable:
        worker.compute(args)
