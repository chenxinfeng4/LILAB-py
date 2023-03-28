# python -m lilab.dannce.p2_dataset_com3d_refine_byback 
# %%
import pickle
import numpy as np
import glob
import os.path as osp
import os
import argparse
from lilab.cvutils_new.parse_name_of_extract_frames import parsename


voxel_origin_pklfiles=[
    # '/home/liying_lab/chenxinfeng/DATA/dannce/data/bw_rat_1280x800x9_2022-10-13_SHANK3_2_voxel_anno_dannce.pkl',
    '/home/liying_lab/chenxinfeng/DATA/dannce/data_SOLO/one_rat_1280x800x9_2023-2-18_young_anno_dannce.pkl',
    #'/home/liying_lab/chenxinfeng/DATA/dannce/data/bw_rat_1280x800x9_2022-4-25_wt_voxel_anno_dannce.pkl',
]


# %%
def ref_com3d(voxel_origin_pklfile):
    pkldata = pickle.load(open(voxel_origin_pklfile, 'rb'))
    outpkldir = osp.join(osp.dirname(voxel_origin_pklfile), 're_com3d')
    os.makedirs(outpkldir, exist_ok=True)
    pklnake = osp.splitext(osp.basename(voxel_origin_pklfile))[0]
    outpkl = pklnake.replace('_anno_dannce','') + '_re_com3d_vol%d' + '_anno_dannce.pkl'

    data_3D = pkldata['data_3D']
    pts3d = data_3D.reshape((data_3D.shape[0], -1, 3))
    com3d = pts3d[:,4,:]
    d = np.linalg.norm((com3d- pkldata['com3d'])[...,:2], axis=-1)
    print('Diff com3d: %.2f (mm)' % np.nanmean(d))
    halflength = np.nanmax(np.linalg.norm(pts3d - com3d[:,None,:], axis=-1), axis=-1)
    bodylength = np.percentile(halflength*2, 90)
    vol_size = np.ceil(bodylength/10 + 1) * 10
    print('vol_size: %.0f (mm)' % np.nanmean(vol_size))

    pkldata['com3d'] = com3d
    pkldata['vol_size'] = vol_size
    pickle.dump(pkldata, open(osp.join(outpkldir, outpkl % pkldata['vol_size']), 'wb'))

    pkldata['vol_size'] = vol_size + 30
    pickle.dump(pkldata, open(osp.join(outpkldir, outpkl % pkldata['vol_size']), 'wb'))


def convert(voxel_origin_pklfiles):
    assert all(osp.exists(spkl) for spkl in voxel_origin_pklfiles)
    for voxel_origin_pklfile in voxel_origin_pklfiles:
        ref_com3d(voxel_origin_pklfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Refine the com3d by the back keypoint")
    parser.add_argument('dannce_pkl', type=str, help='dannce pkl file')
    args = parser.parse_args()
    voxel_origin_pklfiles = [args.dannce_pkl]
    convert(voxel_origin_pklfiles)
