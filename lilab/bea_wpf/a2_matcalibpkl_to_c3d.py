# python -m lilab.bea_wpf.a2_matcalibpkl_to_c3d /A/B/C
# %%
import pickle
import numpy as np
from lilab.bea_wpf.s1_matcalibpkl_to_bea_3d import indmap_matcalib2bea, point_labels
import c3d
import os.path as osp
import argparse
import glob

matcalibpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY50/2023-01-18_16-42-19AbVPAxBwVPA.smoothed_foot.matcalibpkl'


def convert_kpt_3d_to_c3d(kpt_3d_orig, out_c3d_file):
    kpt_3d_bea = kpt_3d_orig[:,:,indmap_matcalib2bea,:]

    nframe, nanimal, nkpt, ndim = kpt_3d_bea.shape

    kpt_3d_bea_full = np.zeros((nframe, nanimal*nkpt, ndim), 'float64')

    for ianimal in range(nanimal):
        kpt_3d_bea_full[:,nkpt*ianimal:nkpt*(ianimal+1),:] = kpt_3d_bea[:,ianimal]

    kpt_3d_bea_full *= 10
    kpt_xy_head = kpt_3d_bea_full[:,[0],:2]
    kpt_xy_head_mean = (np.min(kpt_xy_head, axis=0, keepdims=True) + 
                        np.max(kpt_xy_head, axis=0, keepdims=True))/2
    kpt_3d_bea_full[...,:2] -= kpt_xy_head_mean


    if nanimal==1:
        point_labels_all = point_labels
    elif nanimal==2:
        point_labels_all = point_labels + ['W_'+l for l in point_labels]
    elif nanimal==3:
        point_labels_all = point_labels + ['W_'+l for l in point_labels] + ['D_'+l for l in point_labels]
        print('Warning: nanimal=%d' % nanimal)
    else:
        raise 'Error: nanimal=%d' % nanimal

    kpt_3d_dim5 = np.zeros((kpt_3d_bea_full.shape[0], kpt_3d_bea_full.shape[1], 5), dtype=float)
    kpt_3d_dim5[...,:3] = kpt_3d_bea_full
    writer = c3d.Writer()
    writer.header.frame_rate=30
    writer.set_point_labels(point_labels_all)

    for i in range(kpt_3d_dim5.shape[0]):
        writer.add_frames((kpt_3d_dim5[i], np.array([])), i)

    with open(out_c3d_file, 'wb') as h:
        writer.write(h)
        print('Done convertion to c3d.')


def main(matcalibpkl):
    out_c3d_file = osp.join(osp.dirname(matcalibpkl), osp.basename(matcalibpkl).split('.')[0] + '.c3d')

    data = pickle.load(open(matcalibpkl, 'rb'))
    kpt_3d_orig = data['keypoints_xyz_ba']
    convert_kpt_3d_to_c3d(kpt_3d_orig, out_c3d_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('matcalibpkl_or_dir', type=str)
    args = parser.parse_args()

    if osp.isdir(args.matcalibpkl_or_dir):
        matcalibpkl_l = glob.glob(osp.join(args.matcalibpkl_or_dir, '*.smoothed_foot.matcalibpkl'))
    elif osp.isfile(args.matcalibpkl_or_dir):
        matcalibpkl_l = [args.matcalibpkl_or_dir]
    else:
        raise ValueError('matcalibpkl_or_dir must be a directory or a file')

    for matcalibpkl in matcalibpkl_l:
        main(matcalibpkl)
# %%
