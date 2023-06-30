# python -m lilab.outlier_refine.s4_json_to_danncelabel out.json
import pickle
import argparse
import json
import os.path as osp
import numpy as np
import scipy.io as sio
import tqdm
from lilab.dannce.s1_matcalibpkl2anno import pack_anno_data, pack_calib_data
pkldata_dict = {}

def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    imagebasenames, parsednames = data['imagebasenames'], data['parsednames']
    return imagebasenames, parsednames


def load_pkl(mp4, idrat, iframe):
    if mp4 in pkldata_dict:
        pkldata = pkldata_dict[mp4]
    else:
        pkl_file = osp.splitext(mp4)[0] + '.matcalibpkl'
        pkldata = pickle.load(open(pkl_file, 'rb'))
        pkldata_dict[mp4] = pkldata

    kpt3d = pkldata['keypoints_xyz_ba'][iframe][idrat]
    return kpt3d


def convert(json_file, video_dir):
    imagebasenames, parsednames = load_json(json_file)
    kpt3d = []
    for parsedname in tqdm.tqdm(parsednames):
        mp4, idrat, iframe = parsedname
        mp4 = osp.join(video_dir, mp4)
        idrat = int(idrat)
        iframe = int(iframe)
        kpt3d.append(load_pkl(mp4, idrat, iframe))
    
    kpt3d = np.array(kpt3d)
    ba_poses = pkldata_dict[mp4]['ba_poses']
    out_calibpkl_mat = osp.splitext(json_file)[0] + '.calibpkl.mat'
    out_dict = {}
    out_dict.update(pack_calib_data(ba_poses))
    out_dict.update(pack_anno_data(imagebasenames, kpt3d))
    sio.savemat(out_calibpkl_mat, out_dict)
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', type=str)
    parser.add_argument('--video_dir', type=str, default=None)
    args = parser.parse_args()
    if args.video_dir is None:
        args.video_dir = osp.dirname(osp.dirname(args.json_file))
    convert(args.json_file, args.video_dir)
