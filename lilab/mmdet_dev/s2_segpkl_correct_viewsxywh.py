# python -m lilab.mmdet_dev.s2_segpkl_dilate E:/cxf/mmpose_rat/
# ls *.segpkl | xargs -n 1 -P 0 python -m lilab.mmdet.s2_segpkl_dilate
# %%
import argparse
import os.path as osp
import pickle
import glob
from lilab.cameras_setup import get_view_xywh_wrapper


# class MyWorker(mmap_cuda.Worker):
class MyWorker():
    def compute(self, args):
        pkldata = pickle.load(open(args, 'rb'))
        pkldata['views_xywh'] = get_view_xywh_wrapper(6)
        pickle.dump(pkldata, open(args, 'wb'))


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

    worker = MyWorker()
    for segpkl in segpkl_path:
        worker.compute(segpkl)
