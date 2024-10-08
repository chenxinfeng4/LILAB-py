# python -m lilab.mmdet.s3_segpkl_dilate2videoseg /mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/clips/2022-04-25_16-18-25_bwt_wwt_01time.mp4
import argparse
import os.path as osp
import pickle
import numpy as np
import pycocotools.mask as maskUtils
import cv2
import torch
from tqdm import tqdm
import glob
from lilab.mmlab_scripts.show_pkl_seg_video_fast import imshow_det_bboxes
import lilab.cvutils.map_multiprocess_cuda as mmap_cuda
from lilab.cameras_setup import get_view_xywh_800x600x6 as get_view_xywh
import itertools
import ffmpegcv

video_path = [
    f
    for f in glob.glob(
        "/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/TPH2HETxWT/*.mp4"
    )
    if f[-5] not in "0123456789"
]
pos_views = get_view_xywh()
num_gpus = min([torch.cuda.device_count(), len(get_view_xywh())])


class MyWorker(mmap_cuda.Worker):
    # class MyWorker():
    def compute(self, args):
        (video_in, iview) = args
        self.cuda = getattr(self, "cuda", 0)
        self.id = getattr(self, "id", 0)
        crop_xywh = get_view_xywh()[iview]
        vid = ffmpegcv.VideoReaderNV(video_in, gpu=self.cuda, crop_xywh=crop_xywh)
        segpkl = osp.splitext(video_in)[0] + ".segpkl"
        segpkl_data = pickle.load(open(segpkl, "rb"))
        segdata = segpkl_data["dilate_segdata"][iview]
        video_outb = video_in.replace(".mp4", f"_{iview}_ratblack.mp4")
        video_outw = video_in.replace(".mp4", f"_{iview}_ratwhite.mp4")
        vidout = [
            ffmpegcv.VideoWriterNV(video_outb, fps=vid.fps, gpu=self.cuda),
            ffmpegcv.VideoWriterNV(video_outw, fps=vid.fps, gpu=self.cuda),
        ]
        class_names = ["rat_black", "rat_white"]
        nclass = len(segdata[0])

        for i, (frame, segdata_now) in enumerate(
            zip(
                tqdm(vid, position=int(self.id), desc="worker[{}]".format(self.id)),
                segdata,
            )
        ):
            frame_cuda = torch.from_numpy(frame).cuda().half()
            for iclass in range(nclass):
                mask = segdata_now[1][iclass]
                mask = maskUtils.decode(mask)[:, :, 0]
                # # cuda
                maskcuda = torch.from_numpy(mask).cuda().half()
                frame_cuda_w_mask = frame_cuda * maskcuda[:, :, None]
                frame_w_mask = frame_cuda_w_mask.type(torch.uint8).cpu().numpy()
                # # cpu
                # frame_w_mask = frame * mask[:, :, None]
                vidout[iclass].write(frame_w_mask)

        vid.release()
        for iclass in range(nclass):
            vidout[iclass].release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video_path", type=str, default=None, help="path to video or folder"
    )
    args = parser.parse_args()

    video_path = args.video_path
    assert osp.exists(video_path), "video_path not exists"
    if osp.isfile(video_path):
        video_path = [video_path]
    elif osp.isdir(video_path):
        video_path = glob.glob(osp.join(video_path, "*.mp4"))
        assert len(video_path) > 0, "no video found"
    else:
        raise ValueError("video_path is not a file or folder")

    args_iterable = list(itertools.product(video_path, range(len(pos_views))))

    # init the workers pool
    mmap_cuda.workerpool_init(range(num_gpus), MyWorker)
    mmap_cuda.workerpool_compute_map(args_iterable)
    # worker = MyWorker()
    # worker.compute(args_iterable[0])
