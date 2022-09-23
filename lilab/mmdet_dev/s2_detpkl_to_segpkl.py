# %%python -m lilab.mmdet_dev.s2_detpkl_to_segpkl
# ls *.pkl | xargs -n 1 -P 10 python -m lilab.mmdet_dev.s2_detpkl_to_segpkl
import pickle
import cv2
import pycocotools.mask as maskUtils
import os.path as osp
import numpy as np
import argparse
from lilab.cvutils.map_multiprocess import tqdm

detpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/2022-04-26_15-06-02_bwt_wwt_8.pkl'
# %%
def refine_mask(masks0, masks1):
    if len(masks0) <=1 and len(masks1) <=1:
        pass
    elif len(masks0) == 0 or len(masks1) == 0:
        pass
    elif len(masks0) > 1 or len(masks1) > 1:
        focus_masks, parnter_masks = ((masks0, masks1), (masks1, masks0))
        for focus, parnter in zip(focus_masks, parnter_masks):
            if len(focus) == 1:
                continue
            # case 1: Focus main 接壤 Partner main, Focus sub 接壤 Partner sub。忽略
            partner_main = parnter[0]
            focus_main = focus[0]
            kernel = np.ones((10, 10))
            partner_main_dilate = cv2.dilate(partner_main, kernel, iterations=1)
            focus_main_dilate = cv2.dilate(focus_main, kernel, iterations=1)
            if np.sum(focus_main + partner_main_dilate == 2)>50:
                # Focus main 接壤 Partner main
                for  focus_sub in focus[1:]:
                    focus_sub_unique = focus_sub > partner_main
                    if np.sum(focus_sub_unique + partner_main_dilate==2)>50:
                        # Focus sub 接壤 Partner sub, 保留
                        continue
                    elif np.sum(focus_sub_unique + focus_main_dilate==2)>50:
                        # Focus sub 接壤 Focus main, 保留
                        continue
                    else:
                        # Focus sub 不接壤 Partner sub, 删除
                        focus_sub[:] = 0

            else:
                # Focus main 不接壤 Partner main
                for  focus_sub in focus[1:]:
                    focus_sub_unique = focus_sub > partner_main
                    if np.sum(focus_sub_unique + focus_main_dilate==2)>50:
                        # Focus sub 接壤 Partner sub, 保留
                        continue
                    elif (np.sum(focus_sub_unique + partner_main_dilate==2)>50
                        and np.sum(focus_sub_unique + focus_main_dilate==2)==0):
                        # Focus sub 接壤 Partner sub, 移交给 Partner main
                        partner_main[:] = (partner_main + focus_sub >0).astype(partner_main.dtype)
                        focus_main_dilate = cv2.dilate(focus_main, kernel, iterations=1)
                        focus_sub[:] = 0
                    else:
                        # Focus sub 不接壤 Partner sub, 删除
                        focus_sub[:] = 0

    return masks0, masks1

def mask2resize(mask, shape):
    # mask: CxHxW
    if mask.ndim == 2:
        m1 =  cv2.resize(mask, 
                    shape[::-1], 
                    interpolation=cv2.INTER_NEAREST)
        return m1
    elif mask.ndim == 3:
        m1 =  cv2.resize(mask.transpose(1, 2, 0), 
                    shape[::-1], 
                    interpolation=cv2.INTER_NEAREST)
        if m1.ndim==2:
            m1 = m1[None,:,:]
        else: 
            m1 = m1.transpose(2, 0, 1)
        return m1
    else:
        return mask



def convert(detpkl):
    
    data = pickle.load(open(detpkl, 'rb'))
    outdata = pickle.load(open(detpkl, 'rb'))

    nclass = len(data[0][0])

    shape_base = None
    for frame in data:
        for iclass in range(nclass):
            if frame[1][iclass]:
                shape_base = frame[1][iclass][0]['size']
                break
        if shape_base is not None:
            break
    else:
        raise ValueError('No mask found in the data!')
    
    shape_resize = [400, 400]
    canvas_zero = np.zeros(shape_base)

    maskdcode = lambda x : maskUtils.decode(x).transpose((2,0,1)).astype(np.float32) if x else []
    for frame, frame_out in zip(tqdm(data), outdata):
        masks0 = maskdcode(frame[1][0])
        masks1 = maskdcode(frame[1][1])
        # masks0, masks1 = mask2resize(masks0, shape_resize), mask2resize(masks1, shape_resize)
        masks0, masks1 = refine_mask(masks0, masks1)
        pvals = []
        if len(masks0) and len(masks1):
            masks = np.concatenate([masks0, masks1], axis=0) > 0
        elif len(masks0) == 0 and len(masks1) == 0:
            continue
        elif len(masks0) == 0:
            masks = masks1 > 0
        elif len(masks1) == 0:
            masks = masks0 > 0
        classids = []

        for iclass in range(nclass):
            for boxp, maskzip in zip(frame[0][iclass], frame[1][iclass]):
                pvals.append(boxp[-1])
                classids.append(iclass)

        canvas = canvas_zero.copy()
        pvals = np.array(pvals)
        classids = np.array(classids)
        sort_inds = np.argsort(classids)[::-1] # ascending order
        pvals = pvals[sort_inds]
        masks = masks[sort_inds]
        classids = classids[sort_inds]
        pvals_dict = dict(zip(classids, pvals))
        for mask_mat, iclass in zip(masks, classids):
            canvas[mask_mat] = iclass+1

        # canvas = mask2resize(canvas, shape_base)
        for iclass in range(nclass):
            mask = canvas == (iclass+1)
            if np.sum(mask) <= 4:
                # ignore this mask
                frame_out[0][iclass] = []
            else:
                x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
                pval = pvals_dict[iclass]
                frame_out[0][iclass] = [np.array([x, y, x+w, y+h, pval])]

            frame_out[1][iclass] = maskUtils.encode(
                np.array(mask[:,:,np.newaxis], order='F', dtype=np.uint8))

        # %% save the output
    seg_pkl = osp.splitext(detpkl)[0] + '_seg.pkl'
    pickle.dump(outdata, open(seg_pkl, 'wb'))

    return seg_pkl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('detpkl', 
                        type=str, 
                        help='detection pkl file')
    args = parser.parse_args()
    seg_pkl = convert(args.detpkl)
    print('seg_pkl:', seg_pkl)
