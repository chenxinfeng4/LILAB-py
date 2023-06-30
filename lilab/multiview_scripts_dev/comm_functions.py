# from lilab.multiview_scripts_dev.comm_functions import (
#       box2cs, get_max_preds, get_max_preds_gpu, transform_preds)
import numpy as np
import torch

def box2cs(box, image_size):
    """Encode bbox(x,y,w,h) into (center, scale) without padding.

    Returns:
        tuple: A tuple containing center and scale.
    """
    x, y, w, h = box[:4]

    aspect_ratio = 1. * image_size[0] / image_size[1]
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / 200.0, h * 1.0 / 200.0], dtype=np.float32)
    return center, scale


def get_max_preds(heatmaps:np.ndarray):
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def get_max_preds_gpu(heatmaps:torch.Tensor):
    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = torch.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = torch.amax(heatmaps_reshaped, 2).reshape((N, K, 1))
    idx = idx.cpu().numpy()
    maxvals = maxvals.cpu().numpy()

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def non_max_suppression(heatmap:np.ndarray, num_peaks, w=5) -> np.ndarray:
    peaks = np.zeros((num_peaks, 3), dtype=np.float32)
    for i in range(num_peaks):
        max_index = np.argmax(heatmap)
        peak = np.unravel_index(max_index, heatmap.shape)
        peak_y, peak_x = peak
        peak_v = heatmap[peak_y, peak_x]
        peaks[i] = peak_x, peak_y, peak_v

        y_min = max(0, peak_y - w // 2)
        y_max = min(heatmap.shape[0], peak_y + w // 2 + 1)
        x_min = max(0, peak_x - w // 2)
        x_max = min(heatmap.shape[1], peak_x + w // 2 + 1)
        heatmap[y_min:y_max, x_min:x_max] = 0

    return peaks


def transform_preds(coords, center, scale, output_size, use_udp=False):
    assert coords.shape[-1] == 2
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    scale = scale * 200.0
    output_size = np.array(output_size)
    if use_udp:
        preds = scale / (output_size - 1.0)
    else:
        scale_step = scale / output_size
    preds = coords * scale_step + center - scale * 0.5
    return preds
