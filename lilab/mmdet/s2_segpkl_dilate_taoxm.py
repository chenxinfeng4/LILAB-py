# python -m lilab.mmdet.s2_segpkl_dilate E:/cxf/mmpose_rat/A.segpkl
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


# %%
def dilate(numpy_array):
    kernel = (9, 9)
    stride = 1
    # 1
    # mask = torch.from_numpy(numpy_array).cuda().float()
    # padding = (kernel[0]//2, kernel[1]//2)
    # mask_dilate = torch.nn.functional.max_pool2d(mask, kernel, stride, padding)
    # out_array = mask_dilate.type(torch.uint8).cpu().numpy()
    # out_array = np.asfortranarray(out_array)

    # 2
    mask = torch.from_numpy(numpy_array.transpose((0, 2,1))).cuda().float()
    padding = (kernel[0]//2, kernel[1]//2)
    mask_dilate = torch.nn.functional.max_pool2d(mask, kernel, stride, padding)
    out_array = mask_dilate.type(torch.uint8).cpu().numpy().T
    return out_array


def b_pipeline(q1):
    while True:
        data = q1.get()
        if data is None:
            return
        segout, iclass, mask_dilate = data
        segout[iclass] = mask_util.encode(mask_dilate)


def convert(segpkl, idx=0):
    origin = pickle.load(open(segpkl, 'rb'))
    data = origin['segdata']
    dataout = pickle.load(open(segpkl, 'rb'))['segdata']
    pbar = tqdm.tqdm(total=len(data)*len(data[0])*2, position=idx)
    for iview in range(len(data)):
        for label, labelout in zip(data[iview], dataout[iview]):
            seg, segout = label[1], labelout[1]
            for iclass in range(len(seg)):
                pbar.update(1)
                if len(seg[iclass])==0: continue
                mask = mask_util.decode(seg[iclass]).transpose((2,0,1)) #1xHxW
                mask_dilate = dilate(mask) #HxWx1
                segout[iclass] = mask_util.encode(mask_dilate)

    # save  file
    origin['dilate_segdata'] = dataout
    pickle.dump(origin, open(segpkl, 'wb'))


class MyWorker(mmap_cuda.Worker):
# class MyWorker():
    def compute(self, args):
        self.cuda = getattr(self, 'cuda', 0)
        self.id   = getattr(self, 'id', 0)
        segpkl = args
        with torch.cuda.device(self.cuda):
            return convert(segpkl, idx=self.id)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('segpkl_path', type=str)
    args = argparser.parse_args()
    segpkl_path = args.segpkl_path
    assert osp.exists(segpkl_path), 'segpkl_path not exists'
    if osp.isfile(segpkl_path):
        segpkl_path = [segpkl_path]
    elif osp.isdir(segpkl_path):
        segpkl_path = [f for f in glob.glob(osp.join(segpkl_path, '*.segpkl'))
                        if f[-4] not in '0123456789']
        assert len(segpkl_path) > 0, 'no video found'
    else:
        raise ValueError('segpkl_path is not a file or folder')
    num_gpus = min([torch.cuda.device_count()*2, len(segpkl_path)])

    mmap_cuda.workerpool_init(range(num_gpus), MyWorker)
    mmap_cuda.workerpool_compute_map(segpkl_path)

    # worker = MyWorker()
    # worker.compute(segpkl_path[0])
