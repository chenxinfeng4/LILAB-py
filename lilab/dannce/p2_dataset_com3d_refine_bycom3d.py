# %%
import pickle
import numpy as np
import glob
import os.path as osp
import os
from lilab.cvutils_new.parse_name_of_extract_frames import parsename


video_dir = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/crop9'
# segpkls = glob.glob(video_dir+'/*.segpkl') + glob.glob(video_dir+'/*/*.segpkl')
segpkls = glob.glob(video_dir+'/*.segpkl')
voxel_origin_pklfiles=[
    # '/home/liying_lab/chenxinfeng/DATA/dannce/data/bw_rat_1280x800x9_2022-10-13_SHANK3_2_voxel_anno_dannce.pkl',
    '/home/liying_lab/chenxinfeng/DATA/dannce/data/bw_rat_1280x800x9_2022-4-25_wt_2_voxel_anno_dannce.pkl',
    '/home/liying_lab/chenxinfeng/DATA/dannce/data/bw_rat_1280x800x9_2022-4-25_wt_voxel_anno_dannce.pkl',
]


# %%
def load_com3d(segpkl):
    with open(segpkl, 'rb') as f:
        pkldate = pickle.load(f)
    coms_3d = pkldate['coms_3d'] #(n_x_2rat_x_3)
    return coms_3d


def convert(segpkls, voxel_origin_pklfiles):
    assert all(osp.exists(spkl) for spkl in voxel_origin_pklfiles)
    segpkls_nake = [osp.basename(s) for s in segpkls]
    segpkls_nake2 = [s[:19]+'.segpkl' for s in segpkls_nake]
    assert len(set(segpkls_nake)) == len(segpkls_nake)
    segpkls_coms_3d = {s:load_com3d(segpkl) for s, segpkl in zip(segpkls_nake,segpkls)}
    segpkls_coms_3d.update({s2:segpkls_coms_3d[s] for s, s2 in zip(segpkls_nake, segpkls_nake2)})
    def ref_com3d(voxel_origin_pklfile):
        pkldata = pickle.load(open(voxel_origin_pklfile, 'rb'))
        outpkldir = osp.join(osp.dirname(voxel_origin_pklfile), 're_com3d')
        os.makedirs(outpkldir, exist_ok=True)
        pklnake = osp.splitext(osp.basename(voxel_origin_pklfile))[0]
        outpkl = pklnake.replace('_anno_dannce','') + '_re_com3d_vol%d' + '_anno_dannce.pkl'

        com3d=[]
        for imageName in pkldata['imageNames']:
            vfile, iratkey, iframekey = parsename(imageName)
            segpklkey = osp.splitext(osp.basename(vfile))[0] + '.segpkl'
            com3d.append(segpkls_coms_3d[segpklkey[:19]+'.segpkl'][iframekey, iratkey])

        com3d = np.array(com3d)
        d = np.linalg.norm((com3d- pkldata['com3d'])[...,:2], axis=-1)
        print('Diff com3d: %.2f (mm)' % np.nanmean(d))

        data_3D = pkldata['data_3D']
        pts3d = data_3D.reshape((data_3D.shape[0], -1, 3))
        halflength = np.nanmax(np.linalg.norm(pts3d - com3d[:,None,:], axis=-1), axis=-1)
        bodylength = np.percentile(halflength*2, 90)
        vol_size = np.ceil(bodylength/10 + 1) * 10
        print('vol_size: %.0f (mm)' % np.nanmean(vol_size))

        pkldata['com3d'] = com3d
        pkldata['vol_size'] = vol_size
        pickle.dump(pkldata, open(osp.join(outpkldir, outpkl % pkldata['vol_size']), 'wb'))

        pkldata['vol_size'] = vol_size + 30
        pickle.dump(pkldata, open(osp.join(outpkldir, outpkl % pkldata['vol_size']), 'wb'))


    for voxel_origin_pklfile in voxel_origin_pklfiles:
        ref_com3d(voxel_origin_pklfile)

convert(segpkls, voxel_origin_pklfiles)
