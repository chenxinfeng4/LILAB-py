# Ultralytics YOLO 🚀, AGPL-3.0 license

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import os
import shutil
import glob

from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.files import increment_path


def convert_coco(labels_dir='../coco/annotations/',
                 save_dir='coco_converted/',
                 use_segments=False,
                 use_keypoints=False):
    """
    Converts COCO dataset annotations to a YOLO annotation format  suitable for training YOLO models.

    Args:
        labels_dir (str, optional): Path to directory containing COCO dataset annotation files.
        save_dir (str, optional): Path to directory to save results to.
        use_segments (bool, optional): Whether to include segmentation masks in the output.
        use_keypoints (bool, optional): Whether to include keypoint annotations in the output.
        cls91to80 (bool, optional): Whether to map 91 COCO class IDs to the corresponding 80 COCO class IDs.

    Example:
        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco('../datasets/coco/annotations/', use_segments=True, use_keypoints=False, cls91to80=True)
        ```

    Output:
        Generates output files in the specified output directory.
    """

    # Create dataset directory
    # save_dir = increment_path(save_dir)  # increment if save directory already exists
    assert 'labels' in save_dir, 'save_dir must contain "labels" subdirectory'
    save_images_dir = save_dir.replace('labels', 'images')
    save_label_dir = Path(save_dir).resolve()
    os.makedirs(save_images_dir, exist_ok=True)  # make dir
    save_label_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Import json
    if isinstance(labels_dir, list):
        coco_json_l = [Path(f).resolve() for f in labels_dir]
    elif os.path.isfile(labels_dir):
        coco_json_l = [Path(labels_dir).resolve()]
    else:
        coco_json_l = sorted(Path(labels_dir).resolve().glob('*.json'))

    for json_file in coco_json_l:
        fn = save_label_dir / json_file.stem.replace('instances_', '')  # folder name
        fn.mkdir(parents=True, exist_ok=True)
        with open(json_file) as f:
            data = json.load(f)

        category_ids = sorted([c['id'] for c in data['categories']])
        # Create image dict
        images = {f'{x["id"]:d}': x for x in data['images']}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data['annotations']:
            imgToAnns[ann['image_id']].append(ann)

        # Write labels file
        for img_id, anns in TQDM(imgToAnns.items(), desc=f'Annotations {json_file}'):
            img = images[f'{img_id:d}']
            h, w, f = img['height'], img['width'], img['file_name']

            bboxes = []
            segments = []
            keypoints = []
            for ann in anns:
                if ann['iscrowd']:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = ann['category_id'] - 1  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    if use_segments and ann.get('segmentation') is not None:
                        if len(ann['segmentation']) == 0:
                            segments.append([])
                            continue
                        elif len(ann['segmentation']) > 1:
                            s = merge_multi_segment(ann['segmentation'])
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                        else:
                            s = [j for i in ann['segmentation'] for j in i]  # all segments concatenated
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                        s = [cls] + s
                        segments.append(s)
                    if use_keypoints and ann.get('keypoints') is not None:
                        keypoints.append(box + (np.array(ann['keypoints']).reshape(-1, 3) /
                                                np.array([w, h, 1])).reshape(-1).tolist())

            # Write
            with open((fn / f).with_suffix('.txt'), 'w') as file:
                for i in range(len(bboxes)):
                    if use_keypoints:
                        line = *(keypoints[i]),  # cls, box, keypoints
                    else:
                        line = *(segments[i]
                                 if use_segments and len(segments[i]) > 0 else bboxes[i]),  # cls, box or segments
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')

        # Copy images
        scr_img_dir = os.path.splitext(json_file)[0].replace('_trainval','').replace('_train','').replace('_val','')
        if os.path.exists(scr_img_dir):
            src_imgs = glob.glob(scr_img_dir+'/*.jpg') + glob.glob(scr_img_dir+'/*.png')
            for src_img in src_imgs:
                if os.path.exists(os.path.join(save_images_dir, os.path.basename(src_img))): continue
                shutil.copy2(src_img, save_images_dir)
            LOGGER.info(f'COCO images copied successfully.')
                
    LOGGER.info(f'COCO data converted successfully.\nResults saved to {save_label_dir}')



def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance between two arrays of 2D points.

    Args:
        arr1 (np.array): A NumPy array of shape (N, 2) representing N 2D points.
        arr2 (np.array): A NumPy array of shape (M, 2) representing M 2D points.

    Returns:
        (tuple): A tuple containing the indexes of the points with the shortest distance in arr1 and arr2 respectively.
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """
    Merge multiple segments into one list by connecting the coordinates with the minimum distance between each segment.
    This function connects these coordinates with a thin line to merge all segments into one.

    Args:
        segments (List[List]): Original segmentations in COCO's JSON file.
                               Each element is a list of coordinates, like [segmentation1, segmentation2,...].

    Returns:
        s (List[np.ndarray]): A list of connected segments represented as NumPy arrays.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # Record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # Use two round to connect all the segments
    for k in range(2):
        # Forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # Middle segments have two indexes, reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # Deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s

mask_rcnn_labels_dir = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/data/rats'

coco_json_nake_l = ['bw_rat_1280x800_0425_trainval.json',
               'bw_rat_1280x800x9_20230209_VPA_trainval.json',
               'bw_rat_1280x800_20230828_LTZ_trainval.json',
               'bw_rat_1280x800x9_20221013_small_trainval.json',
               'bw_rat_1280x800_20230525_LTZ_trainval.json']
# coco_json_nake_l = ['bw_rat_1280x800x9_20230608_zyq_trainval.json',
#                     'bw_rat_1280x800_230506_trainval.json',
#                     'bw_rat_1280x800_20231024_WT_trainval.json',
#                     'bw_rat_1280x800_20230810_LTZ_trainval.json',
#                     'bw_rat_1280x800_20230724_trainval.json',
#                     'bw_rat_1280x800_20230701_ranklight_trainval.json',
#                     'bw_rat_1280x800_20230625_WT_trainval.json',
#                     'bw_rat_1280x800_20230524_trainval.json'
#                     ]
            
# coco_json_nake_l = ['bw_rat_1280x800_20230810_LTZ_trainval.json']

coco_json_l = [os.path.join(mask_rcnn_labels_dir, i) for i in coco_json_nake_l]
save_dir = 'data/rats_yolov83/labels/trainval'
convert_coco(coco_json_l, save_dir, use_segments=True)
