import torch
import numpy as np
from warnings import warn
import cv2

import torch.nn.functional as F

# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.



def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale):
    """Get segmentation masks from mask_pred and bboxes.

    Args:
        mask_pred (Tensor or ndarray): shape (n, #class, h, w).
            For single-scale testing, mask_pred is the direct output of
            model, whose type is Tensor, while for multi-scale testing,
            it will be converted to numpy array outside of this method.
        det_bboxes (Tensor): shape (n, 4/5)
        det_labels (Tensor): shape (n, )
        rcnn_test_cfg (dict): rcnn testing config
        ori_shape (Tuple): original image height and width, shape (2,)
        scale_factor(ndarray | Tensor): If ``rescale is True``, box
            coordinates are divided by this scale factor to fit
            ``ori_shape``.
        rescale (bool): If True, the resulting masks will be rescaled to
            ``ori_shape``.

    Returns:
        list[list]: encoded masks. The c-th item in the outer list
            corresponds to the c-th class. Given the c-th outer list, the
            i-th item in that inner list is the mask for the i-th box with
            class label c.

    Example:
        >>> import mmcv
        >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
        >>> N = 7  # N = number of extracted ROIs
        >>> C, H, W = 11, 32, 32
        >>> # Create example instance of FCN Mask Head.
        >>> self = FCNMaskHead(num_classes=C, num_convs=0)
        >>> inputs = torch.rand(N, self.in_channels, H, W)
        >>> mask_pred = self.forward(inputs)
        >>> # Each input is associated with some bounding box
        >>> det_bboxes = torch.Tensor([[1, 1, 42, 42 ]] * N)
        >>> det_labels = torch.randint(0, C, size=(N,))
        >>> rcnn_test_cfg = mmcv.Config({'mask_thr_binary': 0, })
        >>> ori_shape = (H * 4, W * 4)
        >>> scale_factor = torch.FloatTensor((1, 1))
        >>> rescale = False
        >>> # Encoded masks are a list for each category.
        >>> encoded_masks = self.get_seg_masks(
        >>>     mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape,
        >>>     scale_factor, rescale
        >>> )
        >>> assert len(encoded_masks) == C
        >>> assert sum(list(map(len, encoded_masks))) == N
    """
    #mask_pred is tensor
    device = mask_pred.device
    cls_segms = [[] for _ in range(self.num_classes)
                    ]  # BG is not included in num_classes
    bboxes = det_bboxes[:, :4]
    labels = det_labels

    # In most cases, scale_factor should have been
    # converted to Tensor when rescale the bbox
    if not isinstance(scale_factor, torch.Tensor):
        if isinstance(scale_factor, float):
            scale_factor = np.array([scale_factor] * 4)
            warn('Scale_factor should be a Tensor or ndarray '
                    'with shape (4,), float would be deprecated. ')
        assert isinstance(scale_factor, np.ndarray)
        scale_factor = torch.Tensor(scale_factor)

    if rescale:
        img_h, img_w = ori_shape[:2]
        bboxes = bboxes / scale_factor.to(bboxes)
    else:
        w_scale, h_scale = scale_factor[0], scale_factor[1]
        img_h = np.round(ori_shape[0] * h_scale.item()).astype(np.int32)
        img_w = np.round(ori_shape[1] * w_scale.item()).astype(np.int32)

    
    N = len(mask_pred)
    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if device.type == 'cpu':
        # CPU is most efficient when they are pasted one by one with
        # skip_empty=True, so that it performs minimal number of
        # operations.
        num_chunks = N
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)


    threshold = rcnn_test_cfg.mask_thr_binary
    # im_mask = torch.zeros(
    #     N,
    #     img_h,
    #     img_w,
    #     dtype=torch.bool)
    
    im_mask = np.zeros(
        (N,
        img_h,
        img_w),
        dtype=bool)

    if not self.class_agnostic:
        mask_pred = mask_pred[range(N), labels][:, None]

    for inds in chunks:
        # masks_chunk, spatial_inds = _do_paste_mask(
        #     mask_pred[inds],
        #     bboxes[inds],
        #     img_h,
        #     img_w,
        #     skip_empty=device.type == 'cpu')

        masks_chunk, spatial_inds = paste_masks2(
            mask_pred[inds],
            bboxes[inds],
            img_h,
            img_w,
            skip_empty=device.type == 'cpu')
        
        if threshold >= 0:
            masks_chunk = (masks_chunk >= threshold)

        im_mask[(inds, ) + spatial_inds] = masks_chunk

    for i in range(N):
        cls_segms[labels[i]].append(im_mask[i])
    return cls_segms



def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks according to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(
            boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(
            boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    # IsInf op is not supported with ONNX<=1.7.0
    if not torch.onnx.is_in_onnx_export():
        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()
    

def paste_masks2(masks, boxes, img_h, img_w, skip_empty=False):
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    masks = masks.numpy() # 1. 将torch的Tensor张量类型替换为对应的numpy数组。
    boxes = boxes.numpy()
    assert skip_empty
    x0_int, y0_int = np.maximum(
        np.floor(np.min(boxes[:,:2], axis=0)) - 1, 0).astype(np.int64)
    x1_int = np.minimum(np.ceil(np.max(boxes[:,2])) + 1, img_w).astype(np.int64)
    y1_int = np.minimum(np.ceil(np.max(boxes[:,3])) + 1, img_h).astype(np.int64)
    
    N, N2 = masks.shape[:2]
    assert N==1 and N2==1
    mask = np.squeeze(masks)
    mask_resized = cv2.resize(mask, (x1_int - x0_int, y1_int - y0_int))
    return mask_resized[None,...], (slice(y0_int, y1_int), slice(x0_int, x1_int))


def paste_masks(masks, boxes, img_h, img_w, skip_empty=False):
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    masks = masks.numpy() # 1. 将torch的Tensor张量类型替换为对应的numpy数组。
    boxes = boxes.numpy()
    if skip_empty:
        x0_int, y0_int = np.maximum(
            np.floor(np.min(boxes[:,:2], axis=0)) - 1, 0).astype(np.int32)
        x1_int = np.minimum(np.ceil(np.max(boxes[:,2])) + 1, img_w).astype(np.int32)
        y1_int = np.minimum(np.ceil(np.max(boxes[:,3])) + 1, img_h).astype(np.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = np.split(boxes, 4, axis=1)  # each is Nx1

    N = masks.shape[0]

    img_y = np.arange(y0_int, y1_int, dtype=np.float32) + 0.5
    img_x = np.arange(x0_int, x1_int, dtype=np.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    _, H2 = img_y.shape
    _, W2 = img_x.shape
    _, _, H, W = masks.shape
    ind_mask_x = np.clip(line_scale([-1, 1], [0, W-1], img_x), 0, W-1)
    ind_mask_y = np.clip(line_scale([-1, 1], [0, H-1], img_y), 0, H-1)
    ind_mask_x = np.rint(ind_mask_x).astype(int)
    ind_mask_y = np.rint(ind_mask_y).astype(int)

    assert N==1
    # gx = img_x[:, None, :].repeat(img_y.shape[1], axis=1)
    # gy = img_y[:, :, None].repeat(img_x.shape[1], axis=2)
    gx, gy = np.meshgrid(ind_mask_x, ind_mask_y)
    # gx, gy = gx[None, ...], gy[None,...]
    # grid = np.stack([gx, gy], axis=3)

    img_masks = masks[..., gy, gx]
    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()
    

def line_scale(XR, YR, XNow):
    """
    XR: 1x2 ndarray, the [start, end] value of the input range X
    YR: 1x2 ndarray, the [start, end] value of the output range Y
    XNow: n-dimensional ndarray, the input data

    Returns:
    YNow: n-dimensional ndarray, with same size as XNow
    """
    XR = np.array(XR).squeeze()
    YR = np.array(YR).squeeze()
    if not isinstance(XNow, np.ndarray):
        XNow = np.array(XNow)

    assert XR.shape == YR.shape == (2,)

    assert XR[0]!=XR[1], "The values in XR are identical"

    k = (YR[1] - YR[0]) / (XR[1] - XR[0])
    b = (XR[0] * YR[1] - XR[1] * YR[0]) / (XR[0] - XR[1])
    YNow = k * XNow + b
    return YNow