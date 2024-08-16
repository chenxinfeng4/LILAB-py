# python -m lilab.mmdet_dev.s3_segpkl2video E:/cxf/mmpose_rat/A.mp4
# python -m lilab.mmdet_dev.s3_segpkl2video E:/cxf/mmpose_rat/
import argparse
import os.path as osp
import pickle
import numpy as np
import pycocotools._mask as mask_util
import cv2
import torch
from tqdm import tqdm
import glob
from lilab.mmlab_scripts.show_pkl_seg_video_fast import imshow_det_bboxes
import lilab.cvutils.map_multiprocess_cuda as mmap_cuda
from lilab.cameras_setup import get_view_xywh_wrapper
import itertools
import ffmpegcv

pos_views = get_view_xywh_wrapper(6)
video_path = [
    f
    for f in glob.glob(
        "/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/TPH2HETxWT/*.mp4"
    )
    if f[-5] not in "0123456789"
]

num_gpus = min([torch.cuda.device_count(), len(pos_views)])


class MyWorker(mmap_cuda.Worker):
    # class MyWorker():
    def compute(self, args):
        (video_in, iview) = args
        self.cuda = getattr(self, "cuda", 0)
        self.id = getattr(self, "id", 0)
        crop_xywh = pos_views[iview]
        vid = ffmpegcv.VideoReaderNV(video_in, gpu=self.cuda, crop_xywh=crop_xywh)
        pkl_file = video_in.replace(".mp4", f".segpkl")
        data = pickle.load(open(pkl_file, "rb"))["segdata"][iview]
        video_out = video_in.replace(".mp4", f"_{iview}.mp4")
        vidout = ffmpegcv.VideoWriterNV(video_out, fps=vid.fps, gpu=self.cuda)
        class_names = ["rat_black", "rat_white"]

        for i, (label, img) in enumerate(zip(tqdm(data, position=self.id), vid)):
            img = torch.from_numpy(img.astype(np.float32)).to(f"cuda:{self.cuda}")
            bboxes, segms, labels = [], [], []
            for iclass, _ in enumerate(class_names):
                if len(label[0][iclass]) == 0:
                    continue
                bboxes.append(label[0][iclass])  # append numpy.array
                segms.extend(label[1][iclass])  # extend list
                labels.extend([iclass] * len(label[1][iclass]))
            if len(bboxes):
                bboxes = np.concatenate(bboxes)
                labels = np.array(labels, dtype="int")
                masks = mask_util.decode(segms).transpose((2, 0, 1))
                masks = torch.from_numpy(masks).cuda().type(torch.bool)
                img = imshow_det_bboxes(img, bboxes, labels, masks)
            img = img.type(torch.uint8).cpu().numpy()
            img = cv2.putText(
                img, str(i), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
            assert img.shape == (vid.height, vid.width, 3)
            vidout.write(img)

        vidout.release()


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
    # for args in args_iterable:
    #     worker.compute(args)
