import numpy as np
import torch
from typing import Tuple
import cv2

kernel_np = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
orange_HW = (800, 1280)
pvalue_thr = 0.25
mask_sigmoid_thr = 0.9
mask_thr = np.log(mask_sigmoid_thr/(1-mask_sigmoid_thr))


def singleton(outputs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Input: boxes, scores, anchors, strides_vector. type=[torch.Tensor | np.ndarray]
    Output: boxes_, scores_
    ----
    boxes_ : shape = (nbatch,nclass,4)
    scores_ : shape = (nbatch,nclass)
    mask_   : shape = (nbatch,nclass,H,W)
    """
    if isinstance(outputs[0], np.ndarray):
        boxes, scores, maskcoeff, proto = outputs
    elif isinstance(outputs[0], torch.Tensor):
        boxes, scores, maskcoeff, proto = [o.cpu().numpy() for o in outputs]
    else:
        raise TypeError("Unsupported type for outputs: {}".format(type(outputs[0])))
    
    max_inds = np.argmax(scores, axis=1) #(nbatch,nclass)
    scores = np.take_along_axis(scores, max_inds[...,None,:], axis=1)[:,0] #(nbatch,nclass)
    boxes = np.take_along_axis(boxes[:,:,None,:], max_inds[:,None,:,None], axis=1)[:,0] #(nbatch,nclass,4)
    maskcoeff = np.take_along_axis(maskcoeff[:,:,None,:],  max_inds[:,None,:,None], axis=1)[:,0] #(nbatch,nclass,32)
    mask = np.sum(maskcoeff[...,None,None] * proto[:,None,...], axis=2)
    return boxes, scores, mask


def singleton_gpu(outputs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Input: boxes, scores, anchors, strides_vector. type=[torch.Tensor | np.ndarray]
    Output: boxes_, scores_
    ----
    boxes_ : shape = (nbatch,nclass,4)
    scores_ : shape = (nbatch,nclass)
    mask_   : shape = (nbatch,nclass,H,W)
    """
    boxes, scores, maskcoeff, proto = outputs
    max_inds = torch.argmax(scores, axis=1) #(nbatch,nclass)
    scores = torch.take_along_dim(scores, max_inds[...,None,:], dim=1)[:,0] #(nbatch,nclass)
    boxes = torch.take_along_dim(boxes[:,:,None,:], max_inds[:,None,:,None], dim=1)[:,0] #(nbatch,nclass,4)
    maskcoeff = torch.take_along_dim(maskcoeff[:,:,None,:],  max_inds[:,None,:,None], dim=1)[:,0] #(nbatch,nclass,32)
    boxes, scores, maskcoeff, proto = [o.cpu().numpy() for o in [boxes, scores, maskcoeff, proto]]
    mask = np.sum(maskcoeff[...,None,None] * proto[:,None,...], axis=2)
    return boxes, scores, mask


def center_of_mass_cxf(input):
    a_x, a_y = np.sum(input, axis=0, keepdims=True), np.sum(input, axis=1, keepdims=True)
    a_all = np.sum(a_x)
    a_x, a_y = a_x/a_all, a_y/a_all
    grids = np.ogrid[[slice(0, i) for i in input.shape]]
    return np.sum(a_y*grids[0]), np.sum(a_x*grids[1])


def ims_to_com2ds(ims:np.ndarray):
    coms_2d = np.zeros((len(ims), 2), dtype=np.float64)
    for coms_2d_, im_mask in zip(coms_2d, ims):
        assert im_mask.ndim == 2
        if im_mask.size==0 or np.max(im_mask) < 1:
            com_2d = np.ones((2,))+np.nan
        else:
            com_2d = center_of_mass_cxf(im_mask)[::-1]
        coms_2d_[:] = com_2d
    return coms_2d


def refine_mask(outputs:Tuple[np.ndarray, np.ndarray, np.ndarray]) -> \
                        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes, scores, mask = outputs
    NVIEW, NCLASS, *mask_HW = mask.shape
    NN = NVIEW*NCLASS
    ratio_HW_restore = np.array(orange_HW)/np.array(mask_HW)
    mask = mask.reshape(NN, *mask.shape[2:])         #(NN,H,W)
    boxes = boxes.reshape(NN, *boxes.shape[2:])      #(NN,4)
    scores = scores.reshape(NN, *scores.shape[2:])   #(NN,)
    mask_dilate = np.zeros_like(mask, dtype=np.uint8)

    mask = mask>mask_thr
    # coms_real_2d = (boxes[...,[0,1]]+boxes[...,[2,3]])/2.0 * (ratio_HW_restore[None] / 4)
    # coms_real_2d[scores<pvalue_thr] = np.nan

    boxes[...,[0,1]] -= 50
    boxes[...,[0,2]] = np.clip(boxes[...,[0,2]], 0, mask_HW[1]*4, dtype=boxes.dtype)
    boxes[...,[1,3]] = np.clip(boxes[...,[1,3]], 0, mask_HW[0]*4, dtype=boxes.dtype)

    for i, mask_i in enumerate(mask):
        cv2.dilate(mask_i.astype(np.uint8), kernel_np, dst=mask_dilate[i], iterations=1)
    
    mask_dilate[scores<pvalue_thr] = 0
    

    box_for_mask = (boxes // 4).astype(int)
    box_for_mask[scores<pvalue_thr] = 0
    mask_within_roi = np.zeros((NN,), dtype=object)
    for i, ((x1,y1,x2,y2), mask_i) in enumerate(zip(box_for_mask, mask_dilate)):
        # mask_i[:y1,:] = mask_i[y2:] = mask_i[y1:y2,:x1] = mask_i[y1:y2,x2:] = 0
        mask_within_roi[i] = mask_i[y1:y2,x1:x2]

    coms_real_2d = (ims_to_com2ds(mask_within_roi) + boxes[...,[0,1]]/4) * ratio_HW_restore[None]
    scores = scores.reshape(NVIEW, NCLASS, *scores.shape[1:])
    mask_within_roi = mask_within_roi.reshape(NVIEW, NCLASS)
    box_for_mask = box_for_mask.reshape(NVIEW, NCLASS, *box_for_mask.shape[1:])
    coms_real_2d = coms_real_2d.reshape(NVIEW, NCLASS, *coms_real_2d.shape[1:])
    return scores, box_for_mask, mask_within_roi, coms_real_2d
