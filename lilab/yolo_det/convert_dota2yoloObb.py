#%%
from pathlib import Path

import cv2
import numpy as np
import glob
import os
import os.path as osp
import tqdm
import shutil

def convert_dota_to_yolo_obb(dota_root_path: str):
    """
    Converts DOTA dataset annotations to YOLO OBB (Oriented Bounding Box) format.

    The function processes images in the 'train' and 'val' folders of the DOTA dataset. For each image, it reads the
    associated label from the original labels directory and writes new labels in YOLO OBB format to a new directory.

    Args:
        dota_root_path (str): The root directory path of the DOTA dataset.

    Example:
        ```python
        from ultralytics.data.converter import convert_dota_to_yolo_obb

        convert_dota_to_yolo_obb('path/to/DOTA')
        ```

    Notes:
        The directory structure assumed for the DOTA dataset:
            - DOTA
                ├─ images
                │   ├─ train
                │   └─ val
                └─ labels
                    ├─ train_original
                    └─ val_original

        After execution, the function will organize the labels into:
            - DOTA
                └─ labels
                    ├─ train
                    └─ val
    """
    dota_root_path = Path(dota_root_path)

    # Class names to indices mapping
    class_mapping = {
        "mice": 0,
    }

    def convert_label(image_name, image_width, image_height, orig_label_dir, save_dir):
        """Converts a single image's DOTA annotation to YOLO OBB format and saves it to a specified directory."""
        orig_label_path = orig_label_dir / f"{image_name}.txt"
        save_path = save_dir / f"{image_name}.txt"

        with orig_label_path.open("r") as f, save_path.open("w") as g:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                class_name = parts[8]
                class_idx = class_mapping[class_name]
                coords = [float(p) for p in parts[:8]]
                normalized_coords = [
                    coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)
                ]
                formatted_coords = ["{:.6g}".format(coord) for coord in normalized_coords]
                g.write(f"{class_idx} {' '.join(formatted_coords)}\n")

    for phase in ["train", "val"]:
        image_dir = dota_root_path / "images" / phase
        orig_label_dir = dota_root_path / "labels" / f"{phase}_original"
        save_dir = dota_root_path / "labels" / phase

        save_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(image_dir.iterdir())
        for image_path in tqdm.tqdm(image_paths, desc=f"Processing {phase} images"):
            if image_path.suffix not in [".png", ".jpg"]:
                continue
            image_name_without_ext = image_path.stem
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir)


def copy_files(any_labeling_data_path, target_dir):
    img_l = glob.glob(osp.join(any_labeling_data_path, '*.jpg')) + \
            glob.glob(osp.join(any_labeling_data_path, '*.png'))
    txt_l = glob.glob(osp.join(any_labeling_data_path, 'labels/*.txt'))

    nake_name_1 = {osp.splitext(osp.basename(f))[0]:f for f in img_l}
    nake_name_2 = {osp.splitext(osp.basename(f))[0]:f for f in txt_l}
    nake_name = list(set(nake_name_1.keys()) & set(nake_name_2.keys()))

    ratio_val = 0.2
    ind_val = np.random.choice(len(nake_name), int(len(nake_name)*ratio_val), replace=False)
    ind_full = np.zeros(len(nake_name), dtype=bool)
    ind_full[ind_val] = True
    ind_train = np.where(np.logical_not(ind_full))[0]

    images_train = osp.join(target_dir, 'images', 'train')
    images_val = osp.join(target_dir, 'images', 'val')
    labels_train = osp.join(target_dir, 'labels', 'train_original')
    labels_val = osp.join(target_dir, 'labels', 'val_original')
    for folder in [images_train, images_val, labels_train, labels_val]: 
        os.makedirs(folder, exist_ok=True)

    for ind in ind_train:
        shutil.copy(nake_name_1[nake_name[ind]], images_train)
        shutil.copy(nake_name_2[nake_name[ind]], labels_train)

    for ind in ind_val:
        shutil.copy(nake_name_1[nake_name[ind]], images_val)
        shutil.copy(nake_name_2[nake_name[ind]], labels_val)

    out_yaml = f"""
    train: {images_train}
    val: {images_val}
    test: {images_val}

    nc: 1

    names: ['mice']
    """

    out_yaml_path = osp.join(target_dir, 'data.yaml')
    with open(out_yaml_path, 'w') as f: f.write(out_yaml)

#%%
any_labeling_data_path = '/mnt/liying.cibr.ac.cn_Data_Temp/mice_social/outframes/'
target_dir = '/home/liying_lab/chenxinfeng/DATA/ultralytics/data/mice_obb/mice_social_homecage2/'
copy_files(any_labeling_data_path, target_dir)
convert_dota_to_yolo_obb(target_dir)

