# python -m lilab.bea_wpf.s1_matcalibpkl_to_bea_3d A/B/c
# %%
import pickle 
import numpy as np
import c3d
import argparse
import os
import os.path as osp
import glob
import h5py

matcalibpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zyq_to_dzy/20221105_1/multi/2022-11-05_14-57-01_female_non_est_rat1.smoothed_foot'

#%%
indmap_matcalib2bea = [0,1,2,3,6,8,10,12,7,9,11,13,4,5,5,5]
csvheader="""nose,nose,nose,left_ear,left_ear,left_ear,right_ear,right_ear,right_ear,neck,neck,neck,left_front_limb,left_front_limb,left_front_limb,right_front_limb,right_front_limb,right_front_limb,left_hind_limb,left_hind_limb,left_hind_limb,right_hind_limb,right_hind_limb,right_hind_limb,left_front_claw,left_front_claw,left_front_claw,right_front_claw,right_front_claw,right_front_claw,left_hind_claw,left_hind_claw,left_hind_claw,right_hind_claw,right_hind_claw,right_hind_claw,back,back,back,root_tail,root_tail,root_tail,mid_tail,mid_tail,mid_tail,tip_tail,tip_tail,tip_tail
x,y,z,x,y,z,x,y,z,x,y,z,x,y,z,x,y,z,x,y,z,x,y,z,x,y,z,x,y,z,x,y,z,x,y,z,x,y,z,x,y,z,x,y,z,x,y,z
fps:30,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None
"""
point_labels=['nose            ', 'left_ear        ', 'right_ear       ',
        'neck            ', 'left_front_limb ', 'right_front_limb',
        'left_hind_limb  ', 'right_hind_limb ', 'left_front_claw ',
        'right_front_claw', 'left_hind_claw  ', 'right_hind_claw ',
        'back            ', 'root_tail       ', 'mid_tail        ',
        'tip_tail        ']


def convert_to_c3d(kpt_3d, out_c3d_file):
    kpt_3d_dim5 = np.zeros((kpt_3d.shape[0], kpt_3d.shape[1], 5), dtype=float)
    kpt_3d_dim5[...,:3] = kpt_3d
    writer = c3d.Writer()
    writer.header.frame_rate=30
    writer.set_point_labels(point_labels)
    
    for i in range(kpt_3d_dim5.shape[0]):
        writer.add_frames((kpt_3d_dim5[i], np.array([])), i)

    with open(out_c3d_file, 'wb') as h:
        writer.write(h)


def convert_to_csv(kpt_3d, out_csv_file):
    kpt_3d_48 = np.reshape(kpt_3d, (kpt_3d.shape[0], 48))
    with open(out_csv_file, 'w') as f:
        f.write(csvheader)
        np.savetxt(f, kpt_3d_48, delimiter=",", fmt="%.1f")


def convert_to_h5(kpt_3d, out_h5_file):
    kpt_3d_48 = np.reshape(kpt_3d, (kpt_3d.shape[0], 48))
    data3D = kpt_3d_48.astype('float64')
    Bodyparts = np.array([b'nose', b'left_ear', b'right_ear', b'neck', b'left_front_limb',
       b'right_front_limb', b'left_hind_limb', b'right_hind_limb',
       b'left_front_claw', b'right_front_claw', b'left_hind_claw',
       b'right_hind_claw', b'back', b'root_tail', b'mid_tail',
       b'tip_tail'], dtype='|S16')
    FPS = np.array(30, dtype="<i4")

    with h5py.File(out_h5_file, "w") as data_file:
        h5_3Dskeleton = data_file.create_group('3Dskeleton')
        h5_3Dskeleton.create_dataset("FPS", data=FPS)
        h5_3Dskeleton.create_dataset("data3D", data=data3D)
        h5_3Dskeleton.create_dataset("Bodyparts", data=Bodyparts)


def video_name_to_bea_name(video_name):
    video_nake_name = osp.basename(video_name)
    a = video_nake_name
    yy, mm, dd, h_,m_,s_ = a[:4], a[5:7], a[8:10], a[11:13], a[14:16], a[17:19]
    bea_nake_name = f'rec-1-CXF-{yy}{mm}{dd}{h_}{m_}{s_}'

    return bea_nake_name



def main(matcalibpkl, force_ratid):
    projectdir = osp.join(osp.dirname(matcalibpkl), 'BeA_WPF')
    dir_3dskeleton = osp.join(osp.dirname(matcalibpkl), 'BeA_WPF', 'results', '3Dskeleton')
    video_nake_name = osp.basename(matcalibpkl).split('.')[0]
    bea_nake_name = video_name_to_bea_name(video_nake_name)
    out_c3d_file = osp.join(dir_3dskeleton, bea_nake_name+ '_Cali_Data3d.c3d')
    out_csv_file = osp.join(dir_3dskeleton, bea_nake_name+ '_Cali_Data3d.csv')
    os.makedirs(dir_3dskeleton, exist_ok=True)
    
    dir_beaoutputs = osp.join(osp.dirname(matcalibpkl), 'BeA_WPF', 'results', 'BeAOutputs')
    out_h5_file = osp.join(dir_beaoutputs, bea_nake_name + '_results.h5')
    os.makedirs(dir_beaoutputs, exist_ok=True)

    data = pickle.load(open(matcalibpkl, 'rb'))
    if force_ratid is None:
        assert data['keypoints_xyz_ba'].shape[1]==1, 'Should be SOLO rat.'
    
    ratid = 0 if force_ratid is None else {'b':0, 'w':1}[force_ratid]
    kpt_3d_orig = np.squeeze(data['keypoints_xyz_ba'][:,[ratid],:,:])
    assert kpt_3d_orig.ndim==3, 'Should be SOLO rat.'
    assert kpt_3d_orig.shape[1:] == (14,3), 'Keypoint should be 14 x 3.'
    kpt_3d_bea = kpt_3d_orig[:,indmap_matcalib2bea]
    
    convert_to_c3d(kpt_3d_bea, out_c3d_file)
    convert_to_csv(kpt_3d_bea, out_csv_file)
    convert_to_h5(kpt_3d_bea, out_h5_file)
    return projectdir


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

    for matcalibpkl in matcalibpkl_l:
        main(matcalibpkl, args.force_ratid)
        print('Done:', matcalibpkl)
