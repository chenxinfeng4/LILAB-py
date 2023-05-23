# python -m lilab.bea_wpf.p1_matcalibpkl_to_bea_matlab A/B/C
import pickle 
import numpy as np
import argparse
import os
import os.path as osp
import glob
import scipy.io as sio
from lilab.bea_wpf.s1_matcalibpkl_to_bea_3d import indmap_matcalib2bea

matcalibpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zyq_to_dzy/20221105_1/multi/2022-11-05_14-57-01_female_non_est_rat1.smoothed_foot'
fps = 30
downsamplerate = 2

def convert_to_matlab(kpt_3d, out_matlab_file):
    kpt_3d_48 = np.reshape(kpt_3d, (kpt_3d.shape[0], 48))
    coords3d = kpt_3d_48.astype('float64')
    sio.savemat(out_matlab_file, {'coords3d': coords3d})


def create_video_fake(outvfile:str, kpt_3d_bea:np.ndarray):
    fps2 = fps / downsamplerate
    tlen = kpt_3d_bea.shape[0] / fps2
    # use ffmpeg to simulating TV noise for tlen seconds
    cmd = 'ffmpeg -f lavfi -r %f -i color=c=black:s=300x200 -pix_fmt yuv420p -t %f %s' % (fps2, tlen, outvfile)
    os.system(cmd)


def load_kpt_3d_bea(matcalibpkl, force_ratid):
    data = pickle.load(open(matcalibpkl, 'rb'))
    if force_ratid is None:
        assert data['keypoints_xyz_ba'].shape[1]==1, 'Should be SOLO rat.'

    ratid = 0 if force_ratid is None else {'b':0, 'w':1}[force_ratid]
    kpt_3d_orig = np.squeeze(data['keypoints_xyz_ba'][:,ratid,:,:])
    assert kpt_3d_orig.ndim==3, 'Should be SOLO rat.'
    assert kpt_3d_orig.shape[1:] == (14,3), 'Keypoint should be 14 x 3.'
    kpt_3d_bea = kpt_3d_orig[:,indmap_matcalib2bea]
    return kpt_3d_bea


def main(matcalibpkl_l, force_ratid):
    kpt_3d_bea_l = [load_kpt_3d_bea(matcalibpkl, force_ratid) for matcalibpkl in matcalibpkl_l]
    nsample_l = [(matcalibpkl, kpt_3d_bea.shape[0]) for (kpt_3d_bea,matcalibpkl) in zip(kpt_3d_bea_l, matcalibpkl_l)]
    kpt_3d_bea = np.concatenate(kpt_3d_bea_l, axis=0)
    kpt_3d_bea_downsample = kpt_3d_bea[::downsamplerate]

    project_dir = osp.dirname(matcalibpkl_l[0])
    out_matlab_file = osp.join(project_dir, osp.basename(matcalibpkl_l[0]).split('.')[0]+'_merge_coords3d.mat')
    convert_to_matlab(kpt_3d_bea_downsample, out_matlab_file)

    outvfile = osp.join(project_dir, osp.basename(matcalibpkl_l[0]).split('.')[0]+'_merge_fake.avi')
    create_video_fake(outvfile, kpt_3d_bea_downsample)

    outsamplepkl = osp.join(project_dir, osp.basename(matcalibpkl_l[0]).split('.')[0]+'_merge_samplesinfo.pkl')
    pickle.dump({'nsample_l':nsample_l, 'downsample_rate': downsamplerate}, open(outsamplepkl, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('matcalibpkl_or_dir', type=str)
    parser.add_argument('--force-ratid', default=None, type=str)
    args = parser.parse_args()

    if osp.isdir(args.matcalibpkl_or_dir):
        matcalibpkl_l = glob.glob(osp.join(args.matcalibpkl_or_dir, '*smoothed_foot.matcalibpkl'))
        assert matcalibpkl_l, 'No any `smoothed_foot.matcalibpkl` file found in this dir.'
    elif osp.isfile(args.matcalibpkl_or_dir):
        matcalibpkl_l = [args.matcalibpkl_or_dir]
    else:
        raise ValueError('matcalibpkl_or_dir must be a directory or a file')

    assert args.force_ratid in [None, 'b', 'w'], 'force_ratid must be `b` or `w` or None(Solo).'
    main(matcalibpkl_l, args.force_ratid)
