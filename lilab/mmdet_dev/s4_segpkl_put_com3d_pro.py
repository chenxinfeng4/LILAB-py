# python -m lilab.mmdet_dev.s4_segpkl_put_com3d xxx.segpkl --calibpkl xxx.calibpkl
# ls *.segpkl | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s4_segpkl_put_com3d
import pickle
import numpy as np
import argparse
import pycocotools.mask as maskUtils
from scipy.ndimage.measurements import center_of_mass
import tqdm
from lilab.multiview_scripts_dev.s6_calibpkl_predict import CalibPredict
import scipy.signal
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool, Manager

segpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-25_17-08-41_SHANK20_wHetxbKO.segpkl'

calibpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/TPH2-KO-multiview-202201/male/ball/ball.calibpkl'


def center_of_mass_cxf(input):
    a_x, a_y = np.sum(input, axis=0, keepdims=True), np.sum(input, axis=1, keepdims=True)
    a_all = np.sum(a_x)
    a_x, a_y = a_x/a_all, a_y/a_all
    grids = np.ogrid[[slice(0, i) for i in input.shape]]
    return np.sum(a_y*grids[0]), np.sum(a_x*grids[1])


def ims_to_com2ds(ims):
    coms_2d = []
    for im_mask in ims:
        assert im_mask.ndim == 2
        com_2d = center_of_mass_cxf(im_mask)[::-1] if np.max(im_mask) >= 1 else np.ones((2,))+np.nan
        coms_2d.append(com_2d)
    coms_2d = np.array(coms_2d)
    return coms_2d

# %%
def convert(segpkl, calibpkl):
    pkl_data = pickle.load(open(segpkl, 'rb'))
    assert 'coms_3d' not in pkl_data

    views_xywh = pkl_data['views_xywh']
    segdata = pkl_data['dilate_segdata']
    calibPredict = CalibPredict(calibpkl)

    if len(views_xywh)==6:
        mask_original_shape = (600, 800)
    elif len(views_xywh)==10:
        mask_original_shape = (800, 1200)
    else:
        raise NotImplementedError

    mask_real_shape = np.array(segdata[0][0][1][0][0]['size'])
    resize_scale = np.array([[mask_original_shape[1]/mask_real_shape[1], 
                            mask_original_shape[0]/mask_real_shape[0]]])
    nviews = len(views_xywh)
    nframes = len(segdata[0])
    nclass = 2

    global worker
    def worker(iframe):
        coms_real_2d = np.zeros((nviews, nclass, 2))
        mask_ims = [[] for _ in range(nclass)]
        for iview in range(nviews):
            segdata_iview = segdata[iview][iframe]
            for iclass in range(nclass):
                mask = segdata_iview[1][iclass]
                try:
                    mask = maskUtils.decode(mask)[:,:,0]
                except:
                    mask = np.zeros(mask_original_shape)
                    print('mask decode error in {}-{}-{}'.format(iframe, iview, iclass))
                mask_ims[iclass].append(mask)
        for iclass in range(nclass):
            coms_real_2d[:, iclass, :] = ims_to_com2ds(mask_ims[iclass])
        return coms_real_2d

    with Pool(processes=10) as pool:
        # result = pool.map(worker, range(nframes))
        results = list(tqdm.tqdm(pool.imap(worker, range(nframes)), total=nframes))
    
    coms_real_2d = np.array(results).transpose(1, 0, 2, 3)  # nviews, nframes, nclass, 2
    coms_2d = coms_real_2d * resize_scale  # nviews, nframes, nclass, 2
    coms_3d = calibPredict.p2d_to_p3d(coms_2d) # nframes, nclass, 3

    #%%
    coms_3d_smooth = np.zeros_like(coms_3d)
    for is1 in range(coms_3d.shape[1]):
        for is2 in range(coms_3d.shape[2]):
            temp = scipy.signal.medfilt(coms_3d[:,is1,is2], kernel_size = 7)
            temp = gaussian_filter1d(temp, sigma=5)
            coms_3d_smooth[:,is1,is2] = temp

    coms_2d_smooth = calibPredict.p3d_to_p2d(coms_3d_smooth) # nviews, nframes, nclass, 2
    pkl_data.update({
        'coms_3d': coms_3d_smooth,
        'coms_2d': coms_2d_smooth,
        'ba_poses': calibPredict.poses
    })

    # %% dump
    pickle.dump(pkl_data, open(segpkl, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('segpkl', type=str)
    parser.add_argument('--calibpkl', type=str, default=calibpkl)
    args = parser.parse_args()
    convert(args.segpkl, args.calibpkl)
