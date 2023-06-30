# python -m lilab.mmlab_scripts.detpkl_to_segpkl ./data.pkl
# %%
import numpy as np
import mmcv
import pycocotools.mask as maskUtils
import copy
import cv2
import os.path as osp
import argparse


def convert(detpkl):
    data = mmcv.load(detpkl)

    assert data, 'data is empty'
    nclasslist = [len(frame[0]) for frame in data]
    nclassset = set(nclasslist)
    assert len(nclassset) == 1, 'The number of classes in each frame is not the same!'

    nclass = nclasslist[0]

    canvas_zero = None
    for frame in data:
        for iclass in range(nclass):
            if frame[1][iclass]:
                shape = frame[1][iclass][0]['size']
                canvas_zero = np.zeros(shape)
                break
        if canvas_zero is not None:
            break
    else:
        raise ValueError('No mask found in the data!')
    
    # %%
    outdata = copy.deepcopy(data)
    for frame, frame_out in zip(data, outdata):
        # ninstance = [len(i) for i in frame[0]]
        # if max(ninstance) <= 1:
        #     continue

        # multi-instance merge to single instance in each class
        pvals = []
        masks = []
        classids = []

        for iclass in range(nclass):
            for boxp, maskzip in zip(frame[0][iclass], frame[1][iclass]):
                mask_mat = maskUtils.decode(maskzip)
                masks.append(mask_mat)
                pvals.append(boxp[-1])
                classids.append(iclass)

        canvas = canvas_zero.copy()
        pvals = np.array(pvals)
        masks = np.array(masks, dtype = bool)
        classids = np.array(classids)
        sort_inds = np.argsort(pvals) # ascending order
        pvals = pvals[sort_inds]
        masks = masks[sort_inds]
        classids = classids[sort_inds]
        pvals_dict = dict(zip(classids, pvals))
        for mask_mat, iclass in zip(masks, classids):
            canvas[mask_mat] = iclass+1

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
    mmcv.dump(outdata, seg_pkl)

    return seg_pkl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('detpkl', type=str, 
                        help='detection pkl file')
    args = parser.parse_args()
    seg_pkl = convert(args.detpkl)
    print('seg_pkl:', seg_pkl)