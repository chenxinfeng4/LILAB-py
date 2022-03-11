import numpy as np

def getIoU_by_mask(mask_pred, mask_anno, nclass):
    mask_pred = mask_pred.astype(np.uint8)
    mask_anno = mask_anno.astype(np.uint8)
    mask_pred = mask_pred.copy()
    mask_anno = mask_anno.copy()
    ious = []
    for imask in range(1, nclass+1):
        imask_pred = mask_pred == imask
        imask_anno = mask_anno == imask
        intersection = np.logical_and(imask_pred, imask_anno)
        union = np.logical_or(imask_pred, imask_anno)
        iou = np.sum(intersection) / np.sum(union) if np.sum(imask_anno) > 0 else np.nan
        ious.append(iou)
    return ious

def print_iou(ious):
    thrs = np.arange(0.5, 1.0, 0.05)
    mAP = np.mean([np.mean(ious >= thr) for thr in thrs])
    mIoU = np.mean(ious)
    AP50 = np.mean(ious >= 0.5)
    AP75 = np.mean(ious >= 0.75)
    print('mAP: ', mAP)
    print('AP50: ', AP50)
    print('AP75: ', AP75)
    print('mIoU: ', mIoU)
    return mAP, AP50, AP75, mIoU

def summery_stat(predmasks, annomasks):
    # calculate the mAP, AP50, AP75, mIoU according to the ious
    # ious are the iou of each instance
    # mAP is the threshold-based AP, .50:.05:.95 
    # AP50 is the AP of 50% IoU
    # AP75 is the AP of 75% IoU
    # mIoU is the mean IoU.
    # mask: pixel 0 for backgorund, 1..N for objects
    nclass = np.max([np.max(annomask) for annomask in annomasks])
    ious = [getIoU_by_mask(mask_pred, mask_anno, nclass) for (mask_pred, mask_anno) in zip(predmasks, annomasks)]
    ious = np.array(ious).flatten()
    ious = ious[~np.isnan(ious)]
    mAP, AP50, AP75, mIoU = print_iou(ious)


def summery_stat_det(predmasks, annomasks):
    nclass = np.max([np.max(annomask) for annomask in annomasks])
    assert nclass == predmasks.shape[1]
    ious = []
    for iclass in range(nclass):
        predmasks_iclass = predmasks[:, iclass]
        annomasks_iclass = annomasks == iclass+1
        ious_iclass = [getIoU_by_mask(mask_pred, mask_anno, 1) for (mask_pred, mask_anno) 
                        in zip(predmasks_iclass, annomasks_iclass)]
        ious_iclass = np.array(ious_iclass).flatten()
        ious.append(ious_iclass[~np.isnan(ious_iclass)])
    ious = np.concatenate(ious)
    mAP, AP50, AP75, mIoU = print_iou(ious)
