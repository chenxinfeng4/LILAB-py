# python -m lilab.mmdet_dev.s2_segpkl_merge E:/cxf/mmpose_rat/A.mp4
# %%
import pickle
import os.path as osp
import glob
from ffmpegcv.video_info import get_info
import re
import numpy as np
from lilab.cameras_setup import get_view_xywh_wrapper
import argparse

# %%
vfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/ball/2022-04-29_17-58-45_rat.mp4'

def convert(vfile):
    vinfo = get_info(vfile)

    pkl_files = glob.glob(osp.splitext(vfile)[0] + '_*_seg.pkl')
    p = re.compile('.*(\d+)_seg\.pkl$')
    views = [int(p.findall(pkl_file)[0]) for pkl_file in pkl_files]
    assert len(views) == max(views)+1 and min(views) == 0
    views_xywh = get_view_xywh_wrapper(len(views))
    outdata = { 'info': {
                    'vfile': vfile, 
                    'nview': len(views), 
                    'fps': vinfo.fps,
                    'vinfo': vinfo._asdict()},
                'views_xywh': views_xywh,
                'segdata': [] }

    segdata = [[] for _ in range(len(views))]

    for view, pkl_file in zip(views, pkl_files):
        data = pickle.load(open(pkl_file, 'rb'))
        assert isinstance(data, list)
        segdata[view] = data

    outdata['segdata'] = segdata

    # save  file
    outpkl  = osp.splitext(vfile)[0] + '.segpkl'
    pickle.dump(outdata, open(outpkl, 'wb'))
    print('saved to', outpkl)
    return outpkl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str)
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
    
    for vfile in video_path:
        convert(vfile)
