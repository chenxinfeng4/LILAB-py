# python -m lilab.mmpose.s2_mmpose_pkl2matpkl_rat E:/cxf/mmpose_rat/A.mp4
# %%
import pickle
import os.path as osp
import glob
from ffmpegcv.video_info import get_info
import re
import numpy as np
from lilab.cameras_setup import get_view_xywh_1280x800x4 as get_view_xywh
import argparse
from lilab.multiview_scripts_new.s4_matpkl2matcalibpkl import convert as convert2matcalibpkl
# %%
vfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/clips/2022-04-25_16-18-25_bwt_wwt_02time.mp4'
calibpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/ball/2022-04-25ball.calibpkl'

def convert(vfile, calibpkl):
    vinfo = get_info(vfile)

    pkl_files = glob.glob(osp.splitext(vfile)[0] + '_*.kptpkl')
    p = re.compile('.*(\d+)\.kptpkl$')
    views = [int(p.findall(pkl_file)[0]) for pkl_file in pkl_files]
    assert len(views) == max(views)+1 and min(views) == 0

    outdata = { 'info': {
                    'vfile': vfile, 
                    'nview': len(views), 
                    'fps': vinfo.fps,
                    'vinfo': vinfo._asdict()},
                'views_xywh': get_view_xywh(),
                'keypoints': {} }

    keypoints = [[] for _ in range(len(views))]

    for view, pkl_file in zip(views, pkl_files):
        data = pickle.load(open(pkl_file, 'rb'))
        assert isinstance(data, dict) and str(view) in data
        keypoints[view] = data[str(view)]

    keypoints = np.array(keypoints)
    outdata['keypoints'] = keypoints

    # %% save to mat file
    matpkl  = osp.splitext(vfile)[0] + '.matpkl'
    pickle.dump(outdata, open(matpkl, 'wb'))
    print('saved to', matpkl)

    # %% convert to matcalib file
    matcalibpkl = convert2matcalibpkl(matpkl, calibpkl)
    print('calibration to', matcalibpkl)
    return matcalibpkl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vfile', type=str)
    parser.add_argument('calibpkl', type=str)
    args = parser.parse_args()
    convert(args.vfile, args.calibpkl)
