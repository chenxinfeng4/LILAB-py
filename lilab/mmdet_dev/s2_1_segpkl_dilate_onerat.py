# python -m lilab.mmdet_dev.s2_1_segpkl_dilate_onerat E:/cxf/mmpose_rat/A.segpkl
# ls *.segpkl | xargs -n 1 -P 10 python -m lilab.mmdet.s2_segpkl_dilate
# %%
import argparse
from multiprocessing import Process, Queue
import os.path as osp
import pickle
import numpy as np
import pycocotools._mask as mask_util
import torch
import tqdm
import glob
import lilab.cvutils.map_multiprocess_cuda as mmap_cuda
import itertools
import ffmpegcv


def convert(segpkl, idx=0):
    origin = pickle.load(open(segpkl, "rb"))
    data = origin["dilate_segdata"]
    dataout = pickle.load(open(segpkl, "rb"))["dilate_segdata"]
    pbar = tqdm.tqdm(total=len(data) * len(data[0]) * 2, position=idx)
    for iview in range(len(data)):
        for label, labelout in zip(data[iview], dataout[iview]):
            seg, segout = label[1], labelout[1]
            mask = 0
            for iclass in range(len(seg)):
                pbar.update(1)
                if len(seg[iclass]) == 0:
                    continue
                mask = mask_util.decode(seg[iclass]) + mask  # 1xHxW
            if mask is 0:
                mask = np.zeros((512, 512, 1), dtype=np.uint8)
            mask_new = np.array(mask > 0, dtype=np.uint8)
            for iclass in range(len(seg)):
                segout[iclass] = mask_util.encode(mask_new)

    # save  file
    origin["dilate_segdata"] = dataout
    pickle.dump(origin, open(segpkl, "wb"))


class MyWorker(mmap_cuda.Worker):
    # class MyWorker():
    def compute(self, args):
        self.cuda = getattr(self, "cuda", 0)
        self.id = getattr(self, "id", 0)
        segpkl = args
        with torch.cuda.device(self.cuda):
            return convert(segpkl, idx=self.id)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("segpkl_path", type=str)
    args = argparser.parse_args()
    segpkl_path = args.segpkl_path
    assert osp.exists(segpkl_path), "segpkl_path not exists"
    if osp.isfile(segpkl_path):
        segpkl_path = [segpkl_path]
    elif osp.isdir(segpkl_path):
        segpkl_path = [
            f
            for f in glob.glob(osp.join(segpkl_path, "*.segpkl"))
            if f[-4] not in "0123456789"
        ]
        assert len(segpkl_path) > 0, "no video found"
    else:
        raise ValueError("segpkl_path is not a file or folder")
    num_gpus = min([torch.cuda.device_count() * 2, len(segpkl_path)])

    mmap_cuda.workerpool_init(range(num_gpus), MyWorker)
    mmap_cuda.workerpool_compute_map(segpkl_path)

    # worker = MyWorker()
    # worker.compute(segpkl_path[0])
