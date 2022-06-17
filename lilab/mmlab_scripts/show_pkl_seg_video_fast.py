# python -m lilab.mmlab_scripts.show_pkl_seg_video_fast ./data/mmdetection/test_pkl
# %% load packages
import mmcv
import numpy as np
import pycocotools._mask as mask_util
import cv2
from tqdm import tqdm
import os
import torch
import argparse
import glob
import multiprocessing
from multiprocessing import Pool, Value
import ffmpegcv


iPool = Value('i', 0)

def get_mask_colors():
    ncolor = 100
    np.random.seed(42)
    mask_colors = [
        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(ncolor)
    ]
    mask_colors[2:4]=[]
    mask_colors = np.array(mask_colors)
    return mask_colors

default_mask_colors = get_mask_colors()


# %% define the function to draw the mask and the bounding box
def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=13,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray or None): Masks, shaped (n,h,w) or None
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.  Default: 0
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'
        mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        show (bool): Whether to show the image. Default: True
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    mask_colors = default_mask_colors
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        if segms is not None:
            if isinstance(img, torch.Tensor):
                color_mask = mask_colors[labels[i]][:,::-1].copy()
                color_mask = torch.from_numpy(color_mask).to(img.device)
                mask = segms[i]
                img[mask] = img[mask]/2 + color_mask / 2
            else:
                img = img.copy()
                color_mask = mask_colors[labels[i]][:,::-1]
                mask = segms[i].astype(bool)
                img[mask] = img[mask]//2 + color_mask // 2

    return img

# %% load pkl
'''
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

data[iframe][0=bbox][0|1=iclass][ianimal*XYXYP] = numpy.array
data[iframe][1=segm][0|1=iclass][ianimal]   = dict
'''
def pkl_2_video(pkl):
    video_input = pkl.replace('_seg.pkl', '.mp4')
    video_out = pkl.replace('_seg.pkl', '_pkl.mp4')
    data = mmcv.load(pkl)
    v = mmcv.VideoReader(video_input)

    assert len(v) == len(data)
    class_names = ['rat_black', 'rat_white']
    class_nicknames = ['black', 'white']


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ipool = iPool.value
    iPool.value += 1
    video_writer = ffmpegcv.VideoWriterNV(
        video_out, None, v.fps,
        (v.width, v.height),
        gpu = iPool.value)
    # for i, (label, img) in enumerate(zip(tqdm(data, position=ipool), v)):
    for i, (label, img) in enumerate(zip(data, v)):
        bboxes, segms, labels = [], [], []
        for iclass, _ in enumerate(class_names):
            if len(label[0][iclass])==0: continue
            bboxes.append(label[0][iclass]) #append numpy.array
            segms.extend(label[1][iclass])  #extend list
            labels.extend([iclass]*len(label[1][iclass]))
        if len(bboxes):
            bboxes = np.concatenate(bboxes)
            labels = np.array(labels, dtype='int')
            masks = mask_util.decode(segms).transpose((2,0,1))
            img    = imshow_det_bboxes(img, bboxes,labels,masks,class_nicknames, 
                                    show=False, bbox_color='white')
        img = cv2.putText(img, str(i), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if img.shape != (v.height, v.width, 3):
            print(i)
        assert img.shape == (v.height, v.width, 3)
        video_writer.write(img)

    video_writer.release()


# %% __main__
# parse the arguments.
# inputs: pkl_folder
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pkl to video')
    parser.add_argument(
        'pkl_folder',
        help='pkl folder or pkl file',
        type=str)
    args = parser.parse_args()

    # check the existence of the pkl folder
    if os.path.isdir(args.pkl_folder):
        pkl_files = glob.glob(os.path.join(args.pkl_folder, '*_seg.pkl'))
    elif os.path.isfile(args.pkl_folder):
        pkl_files = [args.pkl_folder]
    else:
        raise FileNotFoundError(f'{args.pkl_folder} not found')

    ncpu = multiprocessing.cpu_count()
    maxproc = min([12, ncpu, len(pkl_files)])
    with Pool(processes=maxproc, initargs=(tqdm.get_lock(),),initializer=tqdm.set_lock) as pool:
        pool.map(pkl_2_video, pkl_files)
    