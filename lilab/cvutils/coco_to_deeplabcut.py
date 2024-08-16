# %% imports
import os
import os.path as osp
import pandas as pd
import yaml
import tqdm
import shutil
import argparse
from pycocotools.coco import COCO
from .auto_find_subfolderimages import find_subfolderimages


def main(yamlfile="config.yaml", coco_file="trainval.json"):
    with open(yamlfile, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    bodyparts = cfg["bodyparts"]
    scorer = cfg["scorer"]

    # load the coco_file, and load the keypoints
    coco = COCO(coco_file)
    cats = coco.dataset["categories"]
    assert len(cats) == 1, "Only one category is allowed"
    keypoints = cats[0]["keypoints"]
    img_ids = coco.getImgIds()
    ann_ids = coco.getAnnIds()
    assert len(img_ids) == len(
        ann_ids
    ), "The number of images and annotations are not the same"

    # check bodyparts are equal to keypoints
    assert len(bodyparts) == len(
        keypoints
    ), "The number of bodyparts is not equal to the number of keypoints"
    assert bodyparts == keypoints, "The bodyparts are not the same as the keypoints"

    # %% create new dataframe with multiple headers as 'scorer', 'bodyparts' and 'coords'
    scorers = [scorer]
    coords = ["x", "y"]

    # get the image and annotation by order
    imgs = coco.loadImgs(img_ids)
    imgs_filename = [img["file_name"] for img in imgs]
    assert (
        osp.splitext(imgs_filename[0])[1] == ".png"
    ), "The image file extension is not .png"

    # create multiindex header from the above lists
    header = pd.MultiIndex.from_product(
        [scorers, bodyparts, coords], names=["scorer", "bodyparts", "coords"]
    )
    df_out = pd.DataFrame(columns=header, dtype="float", index=imgs_filename)

    for i, img_id in enumerate(tqdm.tqdm(img_ids)):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        assert len(ann_ids) == 1, "Only one annotation is allowed"
        ann = coco.loadAnns(ann_ids)[0]
        for j, label in enumerate(keypoints):
            point_x, point_y, visible = ann["keypoints"][j * 3 : (j + 1) * 3]
            if visible:
                df_out.loc[imgs_filename[i], (scorer, label, "x")] = point_x
                df_out.loc[imgs_filename[i], (scorer, label, "y")] = point_y

    # replace the '_trainval' with '' of coco_file
    base_name = osp.splitext(osp.basename(coco_file))[0]
    base_name = (
        base_name.replace("_trainval", "").replace("_train", "").replace("_val", "")
    )

    # change the index
    image_fullfiles = [
        "labeled-data/{}/{}".format(base_name, img_filename)
        for img_filename in imgs_filename
    ]
    df_out.index = image_fullfiles

    # save the images to the output folder
    dlc_project_dir = osp.dirname(osp.abspath(yamlfile))
    target_folder = osp.join(dlc_project_dir, "labeled-data", base_name)
    image_fullfiles = find_subfolderimages(coco_file, imgs_filename)
    os.makedirs(target_folder, exist_ok=True)
    for image_file in image_fullfiles:
        target_file = osp.join(target_folder, osp.basename(image_file))
        if not osp.exists(target_file):
            shutil.copy(image_file, target_folder)

    # save the dataframe to the output folder
    h5_file_out = osp.join(target_folder, "CollectedData_{}.h5".format(scorer))
    csv_file_out = osp.join(target_folder, "CollectedData_{}.csv".format(scorer))
    df_out.to_hdf(h5_file_out, "df_with_missing")
    df_out.to_csv(csv_file_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Labelme2deeplabcut")
    parser.add_argument(
        "yaml", type=str, default="config.yaml", help="config.yaml file"
    )
    parser.add_argument(
        "coco_json", type=str, default="PRJECT_trainval.json", help="coco json file"
    )
    args = parser.parse_args()
    main(yamlfile=args.yaml, coco_file=args.coco_json)
