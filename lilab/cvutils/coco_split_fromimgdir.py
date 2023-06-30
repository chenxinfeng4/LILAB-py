import os
import json
import argparse
import funcy
from glob import glob

parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('imgdir', type=str, help='Path to images folder to extract coco json')
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

    prefix =  os.path.splitext(args.annotations)[0]  # without extension
    args_inner = prefix+'_inner.json'
    args_other = prefix+'_other.json'
    image_files = glob(os.path.join(args.imgdir, '*.jpg')) + glob(os.path.join(args.imgdir, '*.png'))
    image_nakefiles = set(funcy.lmap(lambda f: os.path.basename(f), image_files))
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco.get('info', '')
        licenses = ''
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        images_with_annotations = set(funcy.lmap(lambda a: int(a['image_id']), annotations))

        if args.having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        x = funcy.lfilter(lambda i: i['file_name'] in image_nakefiles, images)
        y = funcy.lremove(lambda i: i['file_name'] in image_nakefiles, images)

        save_coco(args_inner, info, licenses, x, filter_annotations(annotations, x), categories)
        save_coco(args_other, info, licenses, y, filter_annotations(annotations, y), categories)

        print("Saved {} entries in {} and {} in {}".format(len(x), args_inner, len(y), args_other))


if __name__ == "__main__":
    main(args)