# python -m lilab.cvutils.labelme_to_cocokeypoints_ball /A/B/C
import argparse
import copy
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np
from labelme import utils
from tqdm import tqdm

bodyparts=['com2d']
TemplateKeypointList = [{'points':[[0,0]],'shape_type':'point'} for i in range(len(bodyparts))]

class AutoId():
    def __init__(self):
        self.id = 1 # start from 1
        self.dict = {} # {NAME: ID}
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
            raise TypeError('item must be int or str')


class Labelme2coco():
    def __init__(self, args):
        self.classname_to_id = {args.class_name: 1}
        self.images = []
        self.annotations = []
        self.categories = []
        self.ann_id = 1 #start from 1
        self.auto_id = AutoId()

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    def _get_box(self, points): #xywh
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

        image['height'], image['width'] = obj['imageHeight'], obj['imageWidth']

        # self.img_id = int(os.path.basename(path).split(".json")[0])
        self.img_id = self.auto_id.query_by_name(os.path.basename(path).split(".json")[0])
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".png")

        return image

    def _annotation(self, bboxes_list, keypoints_list, json_path):
        if len(keypoints_list) != args.join_num * len(bboxes_list):
            print('you loss {} keypoint(s) with file {}'.format(args.join_num * len(bboxes_list) - len(keypoints_list), json_path))
            print('Please check !!!')
            # sys.exit()
        i = 0
        for object in bboxes_list:
            annotation = {}
            keypoints = []
            num_keypoints = 0

            label = object['label']
            bbox = object['points']
            annotation['id'] = self.ann_id
            annotation['image_id'] = self.img_id
            annotation['category_id'] = int(self.classname_to_id[label])
            annotation['iscrowd'] = 0
            
            annotation['bbox'] = self._get_box(bbox)
            annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]

            for keypoint in keypoints_list[i * args.join_num: (i + 1) * args.join_num]:
                point = keypoint['points']
                annotation['keypoints'], num_keypoints = self._get_keypoints(point[0], keypoints, num_keypoints)
            annotation['num_keypoints'] = num_keypoints

            i += 1
            self.ann_id += 1
            self.annotations.append(annotation)

    def _init_categories(self):
        for name, id in self.classname_to_id.items():
            category = {}

            category['supercategory'] = name
            category['id'] = id
            category['name'] = name
            category['keypoints'] = bodyparts
            # category['keypoint'] = [str(i + 1) for i in range(args.join_num)]

            self.categories.append(category)

    def to_coco(self, json_path_list):
        self._init_categories()

        for json_path in tqdm(json_path_list):
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            x,y,w,h = [0, 0, obj['imageWidth'], obj['imageHeight']]
            
            
            bboxes_list, keypoints_list = [{'label':'ball', 'points':[x,y,w,h]}], []
            keypoints_list = copy.copy(TemplateKeypointList)
            for shape in shapes:
                if shape['shape_type'] == 'point':
                    # find the index of the keypoint in bodyparts
                    idx = bodyparts.index(shape['label'])
                    keypoints_list[idx] = shape
                elif shape['shape_type'] == 'rectangle':
                    idx = bodyparts.index(shape['label'])
                    points = shape['points']
                    assert len(points)==2
                    points = [[(points[0][0]+points[1][0])/2, (points[0][1]+points[1][1])/2]]
                    shape['points'] = points
                    shape['shape_type'] = 'point'
                    keypoints_list[idx] = shape
                elif shape['shape_type'] == 'circle':
                    idx = bodyparts.index(shape['label'])
                    shape['shape_type'] = 'point'
                    shape['points'] = [[0, 0]]
                    keypoints_list[idx] = shape
            self._annotation(bboxes_list, keypoints_list, json_path)

        keypoints = {}
        keypoints['info'] = {'description': 'Ball Dataset', 'version': 1.0, 'year': 2020}
        keypoints['license'] = ['Free']
        keypoints['images'] = self.images
        keypoints['annotations'] = self.annotations
        keypoints['categories'] = self.categories
        return keypoints


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="json file path (labelme)", type=str)
    parser.add_argument("--join_num", "--j", help="number of join", type=int, default=1)
    parser.add_argument("--class_name", "--n", help="class name", type=str, default='ball')
    args = parser.parse_args()

    assert len(bodyparts) == args.join_num, "the number of join is not equal to the number of keypoints"

    labelme_path = args.input
    saved_coco_path = args.input + '_trainval.json'

    json_list_path = glob.glob(labelme_path + "/*.json")
    print('{} for trainval'.format(len(json_list_path)))
    print('Start transform please wait ...')

    l2c_train = Labelme2coco(args)
    data_keypoints = l2c_train.to_coco(json_list_path)

    l2c_train.save_coco_json(data_keypoints, saved_coco_path)
