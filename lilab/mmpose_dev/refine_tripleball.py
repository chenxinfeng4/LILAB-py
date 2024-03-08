import numpy as np
from mmpose.core.post_processing import transform_preds
import torch

def non_max_suppression(heatmap:np.ndarray, num_peaks, w=5):
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


arm_nodes = [[0,1],[1,2],[0,2]]
dict_ind_remap = {(0,1,2):(0,1,2),
                  (2,1,0):(0,2,1),
                  (0,2,1):(1,0,2),
                  (2,0,1):(2,0,1),
                  (1,2,0):(1,2,0),
                  (1,0,2):(2,1,0)}


def sort_peaks(peaks_np_shuffled:np.ndarray) -> np.ndarray:
    if len(peaks_np_shuffled)<3:
        return np.zeros((3,3))
    peaks_xy = peaks_np_shuffled[:, :2]
    arm_lens = np.linalg.norm([peaks_xy[n[0]] - peaks_xy[n[1]] 
                                for n in arm_nodes], axis=-1)
    ind_sort = np.argsort(arm_lens)
    ind_sort_remap = np.array(dict_ind_remap[tuple(ind_sort)])
    nodes_xy_sort = peaks_np_shuffled[ind_sort_remap]
    return nodes_xy_sort


def sort_peaks_old(peaks_np):
    if len(peaks_np)<3:
        a = np.zeros((3,3))
        # a[:,:2] = np.nan
        return a
    
    peaks_xy = peaks_np[:, :2]
    
    arm_nodes_set = [set(n) for n in arm_nodes]
    arm_lens = np.linalg.norm([peaks_xy[n[0]] - peaks_xy[n[1]] 
                            for n in arm_nodes], axis=-1)
    ind_min = np.argmin(arm_lens)
    ind_max = np.argmax(arm_lens)
    ind_ball1 = (arm_nodes_set[ind_min] & arm_nodes_set[ind_max]).__iter__().__next__()
    ind_ball2 = (arm_nodes_set[ind_min] - {ind_ball1}).__iter__().__next__()
    ind_ball3 = (arm_nodes_set[ind_max] - {ind_ball1}).__iter__().__next__()
    peaks_np_sort = peaks_np[[ind_ball1, ind_ball2, ind_ball3]]
    return peaks_np_sort


def refine_tripleball(pose_results, returned_outputs):
    heatmap = np.sum(returned_outputs[0]['heatmap'].copy()[0], axis=0)
    peaks = non_max_suppression(heatmap, num_peaks=3, w=10)
    peaks_np = np.array(peaks)
    peaks_np = sort_peaks(peaks_np)

    c = np.array([640,400], dtype=np.float32)
    s = np.array([8, 5], dtype=np.float32)

    peaks_np[:,:2] = transform_preds(
            peaks_np[:,:2], c, s, [512, 320], use_udp=False)
    pose_results[0]['keypoints'] = peaks_np


def get_max_preds_triple(heatmaps:torch.Tensor):
    # sum the C=3 dimension and keep the dimension (N, 1, H, W)
    heatmaps = heatmaps.sum(dim=1, keepdim=True)
    heatmaps = heatmaps.detach().cpu().numpy() # (N, C=1, H=320, W=512)
    N, K, _, W = heatmaps.shape
    peaks = np.zeros((N, K, 3), dtype=np.float32)
    for i in range(N):
        for k in range(K):
            heatmap = heatmaps[i, k]
            peaks[i, k] = non_max_suppression(heatmap, num_peaks=3, w=10)
    preds = peaks[..., :2]
    maxvals = peaks[..., [2]]
    return preds, maxvals

