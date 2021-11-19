#!/usr/bin/python
# !pyinstaller -F coco_split.py -i coco_split.ico
# chenxinfeng
# ------使用方法------
# 直接拖动json到EXE中
#
#python coco_split.py data/鼠关键点/trainval.json
import json
import argparse
import funcy
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('-s', dest='split', type=float, nargs='?', const=0.9,
                    help="A percentage of a split; a number in (0, 1)")
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignore all images without annotations. Keep only these with at least one annotation')

args = parser.parse_args()

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def main(args):
    if args.annotations.endswith('trainval.json'):
        args_train = args.annotations.replace('trainval.json', 'train.json')
        args_test  = args.annotations.replace('trainval.json', 'val.json')
    else:
        prefix =  os.path.splitext(args.annotations)[0] #without extension
        args_train = prefix+'_train.json'
        args_test = prefix+'_val.json'
        
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco.get('info', '')
        licenses = ''
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = set(funcy.lmap(lambda a: int(a['image_id']), annotations))

        if args.having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        x, y = train_test_split(images, train_size=args.split)

        save_coco(args_train, info, licenses, x, filter_annotations(annotations, x), categories)
        save_coco(args_test, info, licenses, y, filter_annotations(annotations, y), categories)

        print("Saved {} entries in {} and {} in {}".format(len(x), args_train, len(y), args_test))


if __name__ == "__main__":
    main(args)