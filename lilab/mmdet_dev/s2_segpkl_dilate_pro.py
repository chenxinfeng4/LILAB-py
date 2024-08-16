# python -m lilab.mmdet_dev.s2_segpkl_dilate E:/cxf/mmpose_rat/
# ls *.segpkl | xargs -n 1 -P 10 python -m lilab.mmdet.s2_segpkl_dilate
# %%
import argparse
import os.path as osp
import pickle
import numpy as np
import pycocotools._mask as mask_util
import torch
import tqdm
import glob
import cv2
import lilab.cvutils.map_multiprocess_cuda as mmap_cuda
from multiprocessing import Pool, Manager

kernel_size = (21, 21)


kernel_np = np.ones(kernel_size, np.uint8)


def dilate_cv(numpy_array):
    out_array = cv2.dilate(numpy_array, kernel_np, iterations=1)
    return out_array[..., None]  # H,W,1


def convert(segpkl, idx=0):
    print("Loadding ...", end="")
    origin = pickle.load(open(segpkl, "rb"))
    print(" Done")
    data = origin["segdata"]
    nview, nframe, nclass = len(data), len(data[0]), len(data[0][0])
    segout = [
        [[[None, None] for _ in range(nclass)] for _ in range(nframe)]
        for _ in range(nview)
    ]

    segdata = data
    mask_original_shape = (800, 1280)

    global worker

    def worker(iview):
        outiview = [[[None, None] for _ in range(nclass)] for _ in range(nframe)]
        if iview == 0:
            pbar = tqdm.tqdm(total=nframe, position=iview, desc=f"[{iview}]")
        else:

            class pbar:
                @staticmethod
                def update(*args, **kargs):
                    pass

        for iframe in range(nframe):
            pbar.update(1)
            segdata_iview = segdata[iview][iframe]
            for iclass in range(nclass):
                mask = segdata_iview[1][iclass]
                try:
                    mask = mask_util.decode(mask)[:, :, 0]
                except:
                    mask = np.zeros(mask_original_shape)
                mask_dilate = dilate_cv(mask)  # HxWx1
                outiview[iframe][iclass] = mask_util.encode(
                    np.asfortranarray(mask_dilate)
                )
        return outiview

    with Pool(processes=nview) as pool:
        # pool = Pool(4)
        # pool.map(worker, range(nview))
        segout = list(pool.imap(worker, range(nview)))

    # save  file
    origin["dilate_segdata"] = segout
    pickle.dump(origin, open(segpkl, "wb"))


# class MyWorker(mmap_cuda.Worker):
class MyWorker:
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
    num_gpus = min([torch.cuda.device_count() * 3, len(segpkl_path)])

    # mmap_cuda.workerpool_init(range(num_gpus), MyWorker)
    # mmap_cuda.workerpool_compute_map(segpkl_path)

    worker = MyWorker()
    worker.compute(segpkl_path[0])
