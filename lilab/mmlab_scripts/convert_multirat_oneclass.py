# %%
import json
import os
from pycocotools.coco import COCO
import copy
import argparse

#coco_file = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/data/rats/coco_BWsiderat_train.json'

def convert_oneclass(coco_file):
    # %% load coco file
    coco = COCO(coco_file)
    cats = coco.loadCats(coco.getCatIds())
    # %% check the categories of the dataset, the name should be like 'rat*' 
    assert all(['rat' in cat['name'].lower() for cat in cats]), 'The dataset should be rat dataset'

    # %% convert
    dataset = copy.deepcopy(coco.dataset)
    dataset['categories'] = [{'id': 1, 'name': 'rat'}]

    # %% set the 'category_id' of each annotation to 1
    for ann in dataset['annotations']:
        ann['category_id'] = 1

    # %% save the converted dataset
    save_coco = os.path.splitext(coco_file)[0] + '_oneclass.json'
    with open(save_coco, 'w') as f:
        json.dump(dataset, f)

    print('Converted dataset saved to {}'.format(save_coco))

# %% main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert multirat dataset to oneclass dataset')
    parser.add_argument('coco_file', type=str, help='The path of the coco file')
    args = parser.parse_args()
    coco_file = args.coco_file
    convert_oneclass(coco_file)