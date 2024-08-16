# folder=`w2l "\\liying.\"`
# folder=/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/behavior/vedio/cpp-1/label_1206
# python -m lilab.cvutils.labelme_to_cocokeypoints_com2d $folder
import sys
import glob
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from labelme import utils
import copy
import cv2
import numpy as np

bodyparts = [
    "Nose",
    "EarL",
    "EarR",
    "Neck",
    "Back",
    "Tail",
    "ForeShoulderL",
    "ForePowL",
    "ForeShoulderR",
    "ForePowR",
    "BackShoulderL",
    "BackPowL",
    "BackShoulderR",
    "BackPowR",
]
TemplateKeypointList = [
    {"points": [[0, 0]], "shape_type": "point"} for i in range(len(bodyparts))
]


class AutoId:
    def __init__(self):
        self.id = 1  # start from 1
        self.dict = {}  # {NAME: ID}

    def query_by_name(self, name):
        if name not in self.dict:
            self.dict[name] = self.id
            self.id += 1
        return self.dict[name]

    def query_by_id(self, id):
        for k, v in self.dict.items():
            if v == id:
                return k
        return None

    def __item__(self, item):
        if isinstance(item, int):
            return self.query_by_id(item)
        elif isinstance(item, str):
            return self.query_by_name[item]
        else:
            raise TypeError("item must be int or str")


class Labelme2coco:
    def __init__(self, args):
        self.classname_to_id = {args.class_name: 1}
        self.images = []
        self.annotations = []
        self.categories = []
        self.ann_id = 1  # start from 1
        self.auto_id = AutoId()

    def save_coco_json(self, instance, save_path):
        json.dump(
            instance,
            open(save_path, "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=1,
        )

    def read_jsonfile(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_box(self, points):  # xywh
        return points

    def _get_keypoints(self, points, keypoints, num_keypoints):
        if points[0] == 0 and points[1] == 0:
            visable = 0
        else:
            visable = 2
            num_keypoints += 1
        keypoints.extend([points[0], points[1], visable])
        return keypoints, num_keypoints

    def _image(self, obj, path):
        image = {}

        img_x = utils.img_b64_to_arr(obj["imageData"])
        image["height"], image["width"] = img_x.shape[:-1]

        # self.img_id = int(os.path.basename(path).split(".json")[0])
        self.img_id = self.auto_id.query_by_name(
            os.path.basename(path).split(".json")[0]
        )
        image["id"] = self.img_id
        image["file_name"] = os.path.basename(path).replace(".json", ".png")

        return image

    def _annotation(self, bboxes_list, keypoints_list, json_path):
        if len(keypoints_list) != args.join_num * len(bboxes_list):
            print(
                "you loss {} keypoint(s) with file {}".format(
                    args.join_num * len(bboxes_list) - len(keypoints_list), json_path
                )
            )
            print("Please check !!!")
            # sys.exit()
        i = 0
        for object in bboxes_list:
            annotation = {}
            keypoints = []
            num_keypoints = 0

            label = object["label"]
            bbox = object["points"]
            annotation["id"] = self.ann_id
            annotation["image_id"] = self.img_id
            annotation["category_id"] = int(self.classname_to_id[label])
            annotation["iscrowd"] = 0

            annotation["bbox"] = self._get_box(bbox)
            annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]

            for keypoint in keypoints_list[i * args.join_num : (i + 1) * args.join_num]:
                point = keypoint["points"]
                annotation["keypoints"], num_keypoints = self._get_keypoints(
                    point[0], keypoints, num_keypoints
                )
            annotation["num_keypoints"] = num_keypoints

            i += 1
            self.ann_id += 1
            self.annotations.append(annotation)

    def _init_categories(self):
        for name, id in self.classname_to_id.items():
            category = {}

            category["supercategory"] = name
            category["id"] = id
            category["name"] = name
            category["keypoints"] = bodyparts
            # category['keypoint'] = [str(i + 1) for i in range(args.join_num)]

            self.categories.append(category)

    def to_coco(self, json_path_list):
        self._init_categories()

        for json_path in tqdm(json_path_list):
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj["shapes"]
            imagePath = obj["imagePath"]
            imagePath = os.path.join(os.path.dirname(json_path), imagePath)
            # read the image as grey color
            img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            imgbin = np.array(img > 0, dtype=np.uint8)
            if np.sum(imgbin) / imgbin.size > 0.5:
                x, y, w, h = 0, 0, img.shape[1] - 1, img.shape[0] - 1
            else:
                # find the xmin, ymin, xmax, ymax of the imgbin
                xmin = np.min(np.where(imgbin.any(axis=0))[0])
                xmax = np.max(np.where(imgbin.any(axis=0))[0])
                ymin = np.min(np.where(imgbin.any(axis=1))[0])
                ymax = np.max(np.where(imgbin.any(axis=1))[0])
                x, y, w, h = xmin, ymin, xmax - xmin, ymax - ymin
                x, y, w, h = int(x), int(y), int(w), int(h)

            bboxes_list, keypoints_list = [{"label": "rat", "points": [x, y, w, h]}], []
            keypoints_list = copy.copy(TemplateKeypointList)
            for shape in shapes:
                if shape["shape_type"] == "point":
                    # find the index of the keypoint in bodyparts
                    idx = bodyparts.index(shape["label"])
                    keypoints_list[idx] = shape

            self._annotation(bboxes_list, keypoints_list, json_path)

        keypoints = {}
        keypoints["info"] = {"description": "Rat Dataset", "version": 1.0, "year": 2020}
        keypoints["license"] = ["Free"]
        keypoints["images"] = self.images
        keypoints["annotations"] = self.annotations
        keypoints["categories"] = self.categories
        return keypoints


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="json file path (labelme)", type=str)
    parser.add_argument(
        "--join_num", "--j", help="number of join", type=int, default=14
    )
    parser.add_argument(
        "--class_name", "--n", help="class name", type=str, default="rat"
    )
    args = parser.parse_args()

    assert (
        len(bodyparts) == args.join_num
    ), "the number of join is not equal to the number of keypoints"

    labelme_path = args.input
    saved_coco_path = args.input + "_trainval.json"

    json_list_path = glob.glob(labelme_path + "/*.json")
    print("{} for trainval".format(len(json_list_path)))
    print("Start transform please wait ...")

    l2c_train = Labelme2coco(args)
    data_keypoints = l2c_train.to_coco(json_list_path)

    l2c_train.save_coco_json(data_keypoints, saved_coco_path)
