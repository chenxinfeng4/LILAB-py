# python -m lilab.mmlab_scripts.show_pkl_seg_video ./data/mmdetection/test_pkl
# %% load packages
import mmcv
import numpy as np
import pycocotools._mask as mask_util
import cv2
from tqdm import tqdm
import copy
import os
from mmdet.core.visualization.image import (EPS, Polygon, PatchCollection, plt, color_val_matplotlib)
import argparse
import glob
import multiprocessing
from multiprocessing import Pool, Value

iPool = Value('i', 0)

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
    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    mask_colors = []
    if labels.shape[0] > 0:
        if mask_color is None:
            # random color
            np.random.seed(42)
            mask_colors = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(len(class_names)+1)
            ]
        else:
            # specify  color
            mask_colors = [
                np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)
            ] * (
                max(labels) + 1)

    bbox_color = color_val_matplotlib(bbox_color)
    text_color = color_val_matplotlib(text_color)

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    polygons = []
    color = []
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(bbox_color)
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        ax.text(
            bbox_int[0],
            bbox_int[1],
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.4,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            fontsize=font_size,
            verticalalignment='top',
            horizontalalignment='left')
        if segms is not None:
            color_mask = mask_colors[labels[i]]
            mask = segms[i].astype(bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

    plt.imshow(img)

    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=1)
    ax.add_collection(p)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img

# %% load pkl
'''
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

data[iframe][0=bbox][0|1=iclass][ianimal*XYXYP] = numpy.array
data[iframe][1=segm][0|1=iclass][ianimal]   = dict
'''
def pkl_2_video(pkl):
    video_input = pkl.replace('.pkl', '.mp4')
    video_out = pkl.replace('.pkl', '_pkl.mp4')
    data = mmcv.load(pkl)
    v = mmcv.VideoReader(video_input)

    assert len(v) == len(data)
    class_names = ['rat_black', 'rat_white']
    class_nicknames = ['black', 'white']


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        video_out, fourcc, v.fps,
        (v.width, v.height))

    ipool = iPool.value
    iPool.value += 1
    for i, (label, img) in enumerate(zip(tqdm(data, position=ipool), v)):
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
        pkl_files = glob.glob(os.path.join(args.pkl_folder, '*.pkl'))
    elif os.path.isfile(args.pkl_folder):
        pkl_files = [args.pkl_folder]
    else:
        raise FileNotFoundError(f'{args.pkl_folder} not found')

    ncpu = multiprocessing.cpu_count()
    maxproc = min([12, ncpu, len(pkl_files)])
    with Pool(processes=maxproc, initargs=(tqdm.get_lock(),),initializer=tqdm.set_lock) as pool:
        pool.map(pkl_2_video, pkl_files)
    