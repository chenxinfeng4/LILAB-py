# conda activate mmdet
# python -m lilab.cvutils.labelme_to_cocokeypoints_tripleball /A/B/C
import lilab.cvutils.labelme_to_cocokeypoints as LCC
from lilab.cvutils.labelme_to_cocokeypoints import Labelme2coco
import argparse
import os
import glob


LCC.bodyparts=['ball1','ball2','ball3']
LCC.TemplateKeypointList = [{'label':i, 'points':[[0,0]],'shape_type':'point'} for i in LCC.bodyparts]
LCC.info = {'description': 'Ripleball Dataset', 'version': 1.0, 'year': 2023}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="json file path (labelme)", type=str)
    args = parser.parse_args()
    args.join_num = len(LCC.bodyparts)
    args.class_name = 'tripleball'

    labelme_path = os.path.abspath(args.input)
    saved_coco_path = labelme_path + '_trainval.json'

    json_list_path = glob.glob(labelme_path + "/*.json")
    print('{} for trainval'.format(len(json_list_path)))
    print('Start transform please wait ...')

    l2c_train = Labelme2coco(args)
    data_keypoints = l2c_train.to_coco(json_list_path)

    l2c_train.save_coco_json(data_keypoints, saved_coco_path)
    