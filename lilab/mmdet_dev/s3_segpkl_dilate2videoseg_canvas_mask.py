# python -m lilab.mmdet_dev.s3_segpkl_dilate2videoseg_canvas_mask /mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/clips/2022-04-25_16-18-25_bwt_wwt_00time.mp4
# python -m lilab.mmdet_dev.s3_segpkl_dilate2videoseg_canvas_mask --disable-dilate /A/B/
import argparse
import os.path as osp
import pickle
import numpy as np
import pycocotools.mask as maskUtils
import cv2
import torch
import copy
from tqdm import tqdm
import glob
import lilab.cvutils.map_multiprocess_cuda as mmap_cuda
import itertools
import ffmpegcv
from lilab.mmdet_dev.canvas_reader import CanvasReader, CanvasReaderThumbnail
from lilab.mmlab_scripts.show_pkl_seg_video_fast import get_mask_colors
from lilab.cameras_setup import get_view_xywh_wrapper as get_view_xywh
video_path = [f for f in glob.glob('/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/TPH2HETxWT/*.mp4')
                if f[-5] not in '0123456789' and 'mask' not in f]


mask_colors = torch.Tensor(get_mask_colors())
nclass = 2

def get_hflips(nviews):
    if nviews==6:
        return [False for _ in range(6)]
    elif nviews==10:
        from lilab.cameras_setup import get_view_hflip
        return [False for _ in range(6)]
        return get_view_hflip()
    else:
        raise NotImplementedError


# class MyWorker(mmap_cuda.Worker):
class MyWorker():
    def compute(self, args):
        video_in, enable_dilate = args
        self.cuda = getattr(self, 'cuda', 0)
        self.id = getattr(self, 'id', 0)
        vid = CanvasReaderThumbnail(video_in, gpu=self.cuda, dilate=enable_dilate)
        video_out = video_in.replace('.mp4', f'_mask.mp4')
        vidout = ffmpegcv.VideoWriterNV(video_out, 
                                        gpu = self.cuda,
                                        codec='h264', 
                                        fps=vid.fps, 
                                        pix_fmt='rgb24')
        for i in tqdm(range(len(vid)), position=int(self.id), 
                                        desc='worker[{}]'.format(self.id)):
            frame_w_mask = vid.read_canvas_mask_img()
            frame_w_mask = cv2.putText(frame_w_mask, str(i), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
            if frame_w_mask is None: break
            vidout.write(frame_w_mask)

        vid.release()
        vidout.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, default=None, help='path to video or folder')
    parser.add_argument('--disable-dilate', action='store_true', help='disable dilate')
    args = parser.parse_args()

    video_path = args.video_path
    assert osp.exists(video_path), 'video_path not exists'
    if osp.isfile(video_path):
        video_path = [video_path]
    elif osp.isdir(video_path):
        video_path = glob.glob(osp.join(video_path, '*.mp4'))
        assert len(video_path) > 0, 'no video found'
    else:
        raise ValueError('video_path is not a file or folder')

    enable_dilate = not args.disable_dilate
    args_iterable = list(itertools.product(video_path, [enable_dilate]))
    num_gpus = min([torch.cuda.device_count()*4, len(args_iterable)])
    # init the workers pool
    # mmap_cuda.workerpool_init(range(num_gpus), MyWorker)
    # mmap_cuda.workerpool_compute_map(args_iterable)

    worker = MyWorker()
    for i in range(len(args_iterable)):
        worker.compute(args_iterable[i])
