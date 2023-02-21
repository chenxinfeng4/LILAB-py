# python -m lilab.dannce.p2_dataset_com3d_refine_byseg xxxx_anno_dannce.pkl
import pickle
import numpy as np
import os.path as osp
import os
import cv2
from lilab.cameras_setup import get_view_xywh_wrapper
from lilab.mmdet_dev.s4_segpkl_put_com3d_pro import ims_to_com2ds
from lilab.mmpose_dev.s3_voxelpkl_2_matcalibpkl import matlab_pose_to_cv2_pose
from lilab.multiview_scripts_dev.s6_calibpkl_predict import CalibPredict
import argparse

voxel_origin_pklfile='/home/liying_lab/chenxinfeng/DATA/dannce/data/bw_rat_1280x800x9_2022-4-25_wt_voxel_anno_dannce.pkl',


def convert(voxel_origin_pklfiles):
    assert all(osp.exists(spkl) for spkl in voxel_origin_pklfiles)
    def ref_com3d(voxel_origin_pklfile):
        pkldata = pickle.load(open(voxel_origin_pklfile, 'rb'))
        outpkldir = osp.join(osp.dirname(voxel_origin_pklfile), 're_com3d')
        os.makedirs(outpkldir, exist_ok=True)
        pklnake = osp.splitext(osp.basename(voxel_origin_pklfile))[0]
        outpkl = pklnake.replace('_anno_dannce','') + '_re_com3d_vol%d' + '_anno_dannce.pkl'
        nview = len(pkldata['imageSize'])
        crop_xywhs = get_view_xywh_wrapper(nview)
        assert np.all(np.array(crop_xywhs)[:,[3,2]] == pkldata['imageSize'])
        ba_poses = matlab_pose_to_cv2_pose(pkldata['camParams'])
        calibPredict = CalibPredict({'ba_poses': ba_poses})
        com3d=[]
        for imageName in pkldata['imageNames']:
            img_canvas = cv2.imread(imageName)
            img_canvas_mask = img_canvas[:,:,0] > 10  # filter background
            ims = [img_canvas_mask[y:y+h, x:x+w] for x,y,w,h in crop_xywhs]
            coms_2d = ims_to_com2ds(ims)
            p3d = calibPredict.p2d_to_p3d(coms_2d)
            com3d.append(p3d)

        com3d = np.array(com3d)
        d = np.linalg.norm((com3d- pkldata['com3d'])[...,:2], axis=-1)
        print('Diff com3d: %.2f (mm)' % np.nanmean(d))

        data_3D = pkldata['data_3D']
        pts3d = data_3D.reshape((data_3D.shape[0], -1, 3))
        halflength = np.nanmax(np.linalg.norm(pts3d - com3d[:,None,:], axis=-1), axis=-1)
        bodylength = np.percentile(halflength*2, 90)
        vol_size = np.ceil(bodylength/10) * 10 + 10
        # vol_size = 170
        print('vol_size: %.0f (mm)' % np.nanmean(vol_size))

        pkldata['com3d'] = com3d
        pkldata['vol_size'] = vol_size
        pickle.dump(pkldata, open(osp.join(outpkldir, outpkl % pkldata['vol_size']), 'wb'))

        pkldata['vol_size'] = vol_size + 30
        pickle.dump(pkldata, open(osp.join(outpkldir, outpkl % pkldata['vol_size']), 'wb'))

    for voxel_origin_pklfile in voxel_origin_pklfiles:
        ref_com3d(voxel_origin_pklfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Voxel dataset com3d refine by centor of mask')
    parser.add_argument('voxel_origin_pklfile', type=str, help='Annotation file')
    args = parser.parse_args()
    voxel_origin_pklfile = args.voxel_origin_pklfile
    assert osp.exists(voxel_origin_pklfile)
    convert([voxel_origin_pklfile])
