# python -m lilab.dannce_single.s1_matcalibpkl_com3d_to_segpkl  xxx.matcalibpkl
# %%
import pickle
import numpy as np
import os.path as osp
import argparse
import scipy.signal
from scipy.ndimage import gaussian_filter1d
from lilab.multiview_scripts_dev.s6_calibpkl_predict import CalibPredict

vfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/SOLO/2023-02-14_13-59-26EwVPA.matcalibpkl'


def convert(vfile):
    data_vfile = pickle.load(open(vfile, 'rb'))

    segpkl = osp.splitext(vfile)[0] + '.segpkl'
    data_segpkl = pickle.load(open(segpkl, 'rb')) if osp.isfile(segpkl) else dict()
    coms_3d = data_vfile['keypoints_xyz_ba']

    #如果np.any(np.isnan(coms_3d))==True
    nan_where = np.where(np.isnan(coms_3d[:,:,0]))[0]
    for i in range(np.isnan(coms_3d[:,:,0]).sum()):
        a = nan_where[i]
        coms_3d[a,:] = coms_3d[a-1,:]

    assert coms_3d.ndim==3 and coms_3d.shape[1:]==(1,3)
    assert not np.any(np.isnan(coms_3d))



    coms_3d_smooth = np.zeros_like(coms_3d)
    for is1 in range(coms_3d.shape[1]):
        for is2 in range(coms_3d.shape[2]):
            temp = scipy.signal.medfilt(coms_3d[:,is1,is2], kernel_size = 7)
            temp = gaussian_filter1d(temp, sigma=5)
            coms_3d_smooth[:,is1,is2] = temp


    calibPredict = CalibPredict(data_vfile)
    coms_2d_smooth = calibPredict.p3d_to_p2d(coms_3d_smooth)

    data_segpkl.update(
        {
            "coms_3d": coms_3d_smooth,
            "coms_2d": coms_2d_smooth,
            "ba_poses": calibPredict.poses,
            'views_xywh': data_vfile['views_xywh']
        }
    )

    pickle.dump(data_segpkl, open(segpkl, "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vfile', type=str, default=vfile)
    args = parser.parse_args()

    assert osp.isfile(args.vfile)
    convert(args.vfile)
