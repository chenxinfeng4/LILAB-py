import numpy as np
import torch
from typing import Tuple
import cv2
from itertools import product
import pycocotools.mask as mask_util
from torch2trt import TRTModule

kernel_np = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)) #832
kernel_np = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) #640
origin_HW = (800, 1280)
pvalue_thr = 0.25
# mask_sigmoid_thr = 0.9
mask_sigmoid_thr = 0.5
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
    mask = torch.sum(maskcoeff[...,None,None] * proto[:,None,...], dim=2)
    boxes, scores, mask = [o.cpu().numpy() for o in [boxes, scores, mask]]
    # mask = np.sum(maskcoeff[...,None,None] * proto[:,None,...], axis=2)
    return boxes, scores, mask


def singleton_gpu_factory(trt_model:TRTModule) -> callable:
    orders = ['bboxes', 'scores', 'maskcoeff', 'proto']
    idx_l = [trt_model.engine.get_binding_index(o)-1 for o in orders]

    def call(outputs:list):
        outputs_reorder = [outputs[i] for i in idx_l]
        return singleton_gpu(outputs_reorder)
    return call


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


# def refine_mask(outputs:Tuple[np.ndarray, np.ndarray, np.ndarray]) -> \
#                         Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     boxes, scores, mask = outputs
#     NVIEW, NCLASS, *mask_HW = mask.shape
#     BN = NVIEW*NCLASS
#     ratio_HW_restore = np.array(origin_HW)/np.array(mask_HW)
#     mask = mask.reshape(BN, *mask.shape[2:])         #(NN,H,W)
#     boxes = boxes.reshape(BN, *boxes.shape[2:])      #(NN,4)
#     scores = scores.reshape(BN, *scores.shape[2:])   #(NN,)
#     mask_dilate = np.zeros_like(mask, dtype=np.uint8)

#     mask = mask>mask_thr
#     # coms_real_2d = (boxes[...,[0,1]]+boxes[...,[2,3]])/2.0 * (ratio_HW_restore[None] / 4)
#     # coms_real_2d[scores<pvalue_thr] = np.nan

#     boxes[...,[0,1]] -= 18
#     boxes[...,[2,3]] += 18
#     boxes[...,[0,2]] = np.clip(boxes[...,[0,2]], 0, mask_HW[1]*4, dtype=boxes.dtype)
#     boxes[...,[1,3]] = np.clip(boxes[...,[1,3]], 0, mask_HW[0]*4, dtype=boxes.dtype)

#     for i, mask_i in enumerate(mask):
#         cv2.dilate(mask_i.astype(np.uint8), kernel_np, dst=mask_dilate[i], iterations=1)
    
#     mask_dilate[scores<pvalue_thr] = 0
#     scores[scores<pvalue_thr] = 0
#     box_for_mask = (boxes // 4).astype(int)
#     box_for_mask[scores<pvalue_thr] = 0
#     mask_within_roi = np.zeros((BN,), dtype=object)
#     for i, ((x1,y1,x2,y2), mask_i) in enumerate(zip(box_for_mask, mask_dilate)):
#         # mask_i[:y1,:] = mask_i[y2:] = mask_i[y1:y2,:x1] = mask_i[y1:y2,x2:] = 0
#         mask_within_roi[i] = mask_i[y1:y2,x1:x2]

#     coms_real_2d = (ims_to_com2ds(mask_within_roi) + boxes[...,[0,1]]/4) * ratio_HW_restore[None]
#     scores = scores.reshape(NVIEW, NCLASS, *scores.shape[1:])
#     mask_within_roi = mask_within_roi.reshape(NVIEW, NCLASS)
#     box_for_mask = box_for_mask.reshape(NVIEW, NCLASS, *box_for_mask.shape[1:])
#     coms_real_2d = coms_real_2d.reshape(NVIEW, NCLASS, *coms_real_2d.shape[1:])
#     return scores, box_for_mask, mask_within_roi, coms_real_2d

def refine_mask(outputs:Tuple[np.ndarray, np.ndarray, np.ndarray]) -> \
                        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes, scores, mask = outputs
    NVIEW, NCLASS, *mask_HW = mask.shape
    BN = NVIEW*NCLASS
    ratio_HW_restore = np.array(origin_HW)/np.array(mask_HW)
    mask = mask.reshape(BN, *mask.shape[2:])         #(NN,H,W)
    boxes = boxes.reshape(BN, *boxes.shape[2:])      #(NN,4)
    scores = scores.reshape(BN, *scores.shape[2:])   #(NN,)
    mask = mask>mask_thr

    # boxes: box_xyxy in feature input size
    boxes[...,[0,1]] -= 15
    boxes[...,[2,3]] += 15
    boxes[...,[0,2]] = np.clip(boxes[...,[0,2]], 0, mask_HW[1]*4, dtype=boxes.dtype)
    boxes[...,[1,3]] = np.clip(boxes[...,[1,3]], 0, mask_HW[0]*4, dtype=boxes.dtype)
    box_for_mask = (boxes // 4).astype(int)
    box_for_mask[scores<pvalue_thr] = -1
    scores[scores<pvalue_thr] = 0
    box_for_image = (box_for_mask * [*ratio_HW_restore, *ratio_HW_restore][::-1]).astype(int)
    box_for_image[...,[0,2]] = np.clip(box_for_image[...,[0,2]], 0, origin_HW[1]*4).astype(int)
    box_for_image[...,[1,3]] = np.clip(box_for_image[...,[1,3]], 0, origin_HW[0]*4).astype(int)
    mask_within_roi = np.zeros((BN,), dtype=object)
    mask_orign_within_roi = np.zeros((BN,), dtype=object)
    for i, (score, mask_i, box_i, box_image_i) in enumerate(zip(scores, mask, box_for_mask, box_for_image)):
        x1a, y1a, x2a, y2a = box_i
        x1b, y1b, x2b, y2b = box_image_i
        mask_within_roi[i] = mask_i_roi = np.array(mask_i[y1a:y2a, x1a:x2a], dtype=np.uint8)
        if score<=0: continue
        mask_dilate_roi = cv2.dilate(mask_i_roi, kernel_np, iterations=1)
        mask_orign_within_roi[i] = cv2.resize(mask_dilate_roi, (x2b-x1b, y2b-y1b), interpolation=cv2.INTER_NEAREST)

    # box_for_mask: box_xyxy in graph output mask size
    coms_real_2d = (ims_to_com2ds(mask_within_roi) + boxes[...,[0,1]]/4) * ratio_HW_restore[None]
    coms_real_2d = coms_real_2d.reshape(NVIEW, NCLASS, *coms_real_2d.shape[1:])
    coms_real_2d[np.isnan(coms_real_2d[:,:,0])] = origin_HW[1]/2, origin_HW[0]/2

    scores = scores.reshape(NVIEW, NCLASS)
    box_for_mask = box_for_mask.reshape(NVIEW, NCLASS, *box_for_mask.shape[-1:])
    mask_orign_within_roi = mask_orign_within_roi.reshape(NVIEW, NCLASS)

    # box_for_mask_orign: box_xyxy in image size
    box_for_mask_orign = box_for_mask * [*ratio_HW_restore, *ratio_HW_restore][::-1]
    box_for_mask_orign = box_for_mask_orign.astype(int)
    return scores, box_for_mask_orign, mask_orign_within_roi, coms_real_2d


def refine_mask2(outputs:Tuple[np.ndarray, np.ndarray, np.ndarray]) -> \
                         Tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes, scores, mask = outputs
    NVIEW, NCLASS, *mask_HW = mask.shape
    BN = NVIEW*NCLASS
    ratio_HW_restore = np.array(origin_HW)/np.array(mask_HW)
    mask = mask.reshape(BN, *mask.shape[2:])         #(NN,H,W)
    boxes = boxes.reshape(BN, *boxes.shape[2:])      #(NN,4)
    scores = scores.reshape(BN, *scores.shape[2:])   #(NN,)
    mask_dilate = np.zeros_like(mask, dtype=np.uint8)
    mask = mask>mask_thr

    # boxes: box_xyxy in feature input size
    boxes[...,[0,1]] -= 15
    boxes[...,[2,3]] += 15
    boxes[...,[0,2]] = np.clip(boxes[...,[0,2]], 0, mask_HW[1]*4, dtype=boxes.dtype)
    boxes[...,[1,3]] = np.clip(boxes[...,[1,3]], 0, mask_HW[0]*4, dtype=boxes.dtype)

    for i, mask_i in enumerate(mask):
        # cv2.dilate(mask_i.astype(np.uint8), kernel_np, dst=mask_dilate[i], iterations=1)
        mask_dilate[i] = cv2.dilate(mask_i.astype(np.uint8), kernel_np, iterations=1)
    mask[:] = 0

    mask_dilate[scores<pvalue_thr] = 0
    scores[scores<pvalue_thr] = 0
    # box_for_mask: box_xyxy in graph output mask size
    box_for_mask = (boxes // 4).astype(int)
    box_for_mask[scores<pvalue_thr] = 0
    mask_within_roi = np.zeros((BN,), dtype=object)
    mask_origin_HW = np.zeros((BN, *origin_HW), dtype=np.uint8)
    for i, ((x1,y1,x2,y2), mask_i) in enumerate(zip(box_for_mask, mask_dilate)):
        mask_within_roi[i] = mask_i[y1:y2,x1:x2]
        if (x1,y1,x2,y2)==(0,0,0,0): continue
        mask_i[:y1,:] = mask_i[y2:,:] = mask_i[y1:y2,:x1] = mask_i[y1:y2,x2:] = 0
        cv2.resize(mask_i, origin_HW[::-1], interpolation=cv2.INTER_NEAREST, dst=mask_origin_HW[i])

    coms_real_2d = (ims_to_com2ds(mask_within_roi) + boxes[...,[0,1]]/4) * ratio_HW_restore[None]
    coms_real_2d = coms_real_2d.reshape(NVIEW, NCLASS, *coms_real_2d.shape[1:])
    coms_real_2d[np.isnan(coms_real_2d[:,:,0])] = origin_HW[1]/2, origin_HW[0]/2

    scores = scores.reshape(NVIEW, NCLASS)
    mask_origin_HW = mask_origin_HW.reshape(NVIEW, NCLASS, *mask_origin_HW.shape[-2:])
    box_for_mask = box_for_mask.reshape(NVIEW, NCLASS, *box_for_mask.shape[-1:])

    # box_for_mask_orign: box_xyxy in image size
    box_for_mask_orign = box_for_mask * [*ratio_HW_restore, *ratio_HW_restore][::-1]
    box_xyxyp = np.concatenate((box_for_mask_orign, scores[...,None]), axis=-1)
    
    # encode mask
    mask_enc = np.empty((NVIEW, NCLASS), dtype=object)
    for iview, iclass in product(range(NVIEW), range(NCLASS)):
        mask_enc[iview, iclass] = mask_util.encode(
                 np.array(mask_origin_HW[iview, iclass],
                          order='F', dtype=np.uint8))
          
    return box_xyxyp, mask_enc, coms_real_2d



def refine_mask3(outputs:Tuple[np.ndarray, np.ndarray, np.ndarray]) -> \
                         Tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes, scores, mask = outputs
    NVIEW, NCLASS, *mask_HW = mask.shape
    BN = NVIEW*NCLASS
    ratio_HW_restore = np.array(origin_HW)/np.array(mask_HW)
    mask = mask.reshape(BN, *mask.shape[2:])         #(NN,H,W)
    boxes = boxes.reshape(BN, *boxes.shape[2:])      #(NN,4)
    scores = scores.reshape(BN, *scores.shape[2:])   #(NN,)
    mask_origin_HW = np.zeros((BN, *origin_HW), dtype=np.uint8)
    mask = mask>mask_thr

    # boxes: box_xyxy in feature input size
    boxes[...,[0,1]] -= 15
    boxes[...,[2,3]] += 15
    boxes[...,[0,2]] = np.clip(boxes[...,[0,2]], 0, mask_HW[1]*4, dtype=boxes.dtype)
    boxes[...,[1,3]] = np.clip(boxes[...,[1,3]], 0, mask_HW[0]*4, dtype=boxes.dtype)
    box_for_mask = (boxes // 4).astype(int)
    box_for_mask[scores<pvalue_thr] = 0
    scores[scores<pvalue_thr] = 0
    box_for_image = (box_for_mask * [*ratio_HW_restore, *ratio_HW_restore][::-1]).astype(int)
    box_for_image[...,[0,2]] = np.clip(box_for_image[...,[0,2]], 0, origin_HW[1]*4).astype(int)
    box_for_image[...,[1,3]] = np.clip(box_for_image[...,[1,3]], 0, origin_HW[0]*4).astype(int)
    mask_within_roi = np.zeros((BN,), dtype=object)

    for i, (mask_i, mask_image_i, box_i, box_image_i) in enumerate(zip(mask, mask_origin_HW, box_for_mask, box_for_image)):
        x1a, y1a, x2a, y2a = box_i
        x1b, y1b, x2b, y2b = box_image_i
        mask_within_roi[i] = mask_i_roi = np.array(mask_i[y1a:y2a, x1a:x2a], dtype=np.uint8)
        if (x1b,y1b,x2b,y2b)==(0,0,0,0): continue
        mask_dilate_roi = cv2.dilate(mask_i_roi, kernel_np, iterations=1)
        mask_dilate_roi_image = cv2.resize(mask_dilate_roi, (x2b-x1b, y2b-y1b), interpolation=cv2.INTER_NEAREST, dst=mask_origin_HW[i])
        mask_image_i[y1b:y2b, x1b:x2b] = mask_dilate_roi_image
    
    # box_for_mask: box_xyxy in graph output mask size
    coms_real_2d = (ims_to_com2ds(mask_within_roi) + boxes[...,[0,1]]/4) * ratio_HW_restore[None]
    coms_real_2d = coms_real_2d.reshape(NVIEW, NCLASS, *coms_real_2d.shape[1:])
    coms_real_2d[np.isnan(coms_real_2d[:,:,0])] = origin_HW[1]/2, origin_HW[0]/2

    scores = scores.reshape(NVIEW, NCLASS)
    mask_origin_HW = mask_origin_HW.reshape(NVIEW, NCLASS, *mask_origin_HW.shape[-2:])
    box_for_mask = box_for_mask.reshape(NVIEW, NCLASS, *box_for_mask.shape[-1:])

    # box_for_mask_orign: box_xyxy in image size
    box_for_mask_orign = box_for_mask * [*ratio_HW_restore, *ratio_HW_restore][::-1]
    box_xyxyp = np.concatenate((box_for_mask_orign, scores[...,None]), axis=-1)
    
    # encode mask
    mask_enc = np.empty((NVIEW, NCLASS), dtype=object)
    for iview, iclass in product(range(NVIEW), range(NCLASS)):
        mask_enc[iview, iclass] = mask_util.encode(
                 np.array(mask_origin_HW[iview, iclass],
                          order='F', dtype=np.uint8))
          
    return box_xyxyp, mask_enc, coms_real_2d
