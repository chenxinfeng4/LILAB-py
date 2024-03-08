# conda activate DEEPLABCUT
# python -m lilab.cvutils.labelme_to_coco LABELME_DIR

# %% import
import labelme2coco
import os
import sys
import json
import sys, os
import datetime
from pycocotools.coco import COCO
# set directory that contains labelme annotations and image files

reclassid = [{'rat_black':1, 'rat_white':2}, 
             {'rat_black':1, 'rat_white':2, 'rat_dot':3},
             {'rat_dot':1, 'rat_white':2},
             {'rat':1},
             {'ratface':1}]

# %% functions
def conver_reclassid(cocojson_file, cocojson_file_out):
    with open(cocojson_file, 'r') as f:
        anno_data = json.load(f)
    anno_cats = anno_data['categories']
    anno_annos = anno_data['annotations']
    cat_to_id = {cat['name']:cat['id'] for cat in anno_cats}
    catnames = set(cat_to_id.keys())
    ref_catnames = [set(c.keys()) for c in reclassid]
    if catnames not in ref_catnames:
        print('new categories:', catnames)
        return
    
    index = ref_catnames.index(catnames)
    ref_cat_to_id = reclassid[index]
    if ref_cat_to_id == cat_to_id:
        return

    oldid_to_newid = {oldid:ref_cat_to_id[name] for name, oldid in cat_to_id.items()}

    # revise cat id
    for anno in anno_cats:
        anno['id'] = oldid_to_newid[anno['id']]

    fun_id = lambda x: x['id']
    anno_cats.sort(key=fun_id)

    for anno in anno_annos:
        anno['category_id'] = oldid_to_newid[anno['category_id']]

    # save file
    with open(cocojson_file_out, 'w') as f:
        json.dump(anno_data, f, indent=2)


def merge_multparts(anno, anno_out):
    coco = COCO(anno)
    dataout = dict()
    dataout['categories'] = coco.dataset['categories']
    dataout['images'] = []
    dataout['annotations'] = []
    dataout['info'] = dict(
        description = 'labelme2posecoco', 
        version = '1.0', 
        year = datetime.datetime.today().year,
        date_created = datetime.datetime.today().strftime('%Y/%m/%d'))
    dataout['licences'] = ''
    dataout['images'] = coco.dataset['images']
    # %%
    annoid =0
    imgids = coco.getImgIds()
    catids = list(coco.cats.keys())
    for image_id in imgids:
        anns = coco.imgToAnns[image_id]
        allrat_seg = {catid:[] for catid in catids}

        for iann in anns:
            catid = iann['category_id']
            allrat_seg[catid] += iann['segmentation']

        for catid, segs in allrat_seg.items():
            if segs == []: continue
            segs_1d = [j for sub in segs for j in sub]
            x_min, y_min = min(segs_1d[::2]), min(segs_1d[1::2])
            x_max, y_max = max(segs_1d[::2]), max(segs_1d[1::2])
            bbox = [int(i) for i in [x_min, y_min, x_max-x_min, y_max-y_min]]
            annotations0 = dict(
                id = annoid,
                bbox = bbox,
                iscrowd = 0,
                area = bbox[2]*bbox[3],
                category_id = catid,
                image_id = image_id,
                segmentation = segs
            )
            dataout['annotations'].append(annotations0)
            annoid+=1

    with open(anno_out, 'w') as f:
        json.dump(dataout, f, indent=2)


def resort(cocojson_file, out_coco_file):
    # load coco json
    with open(cocojson_file, 'r') as f:
        anno_data = json.load(f)

    # get image id to new image id
    anno_images = anno_data['images']
    anno_annos = anno_data['annotations']
    fun_filename = lambda x: x['file_name']
    anno_images.sort(key=fun_filename)
    oldids =  [x['id'] for x in anno_images]
    oldid_to_newid = {oldid:iframe+1 for iframe, oldid in enumerate(oldids)}

    # revise image id
    for anno in anno_annos:
        anno['image_id'] = oldid_to_newid[anno['image_id']]

    for anno in anno_images:
        anno['id'] = oldid_to_newid[anno['id']]

    # save file
    with open(out_coco_file, 'w') as f:
        json.dump(anno_data, f, indent=2)


def converter(labelme_folder):
    parent_folder, cur_folder = os.path.split(os.path.abspath(labelme_folder))
    
    # set path for coco json to be saved
    save_json_path = os.path.join(parent_folder, cur_folder+"_trainval.json")

    # convert labelme annotations to coco
    labelme2coco.convert(labelme_folder, save_json_path)
    # merge multiple parts
    merge_multparts(save_json_path, save_json_path)
    # resort
    resort(save_json_path, save_json_path)
    # convert reclassid
    conver_reclassid(save_json_path, save_json_path)


if __name__ == '__main__':
    n = len(sys.argv)
    if n == 1:
        folder = input("Select a folder >> ")
        if folder == None:
            exit()
        else:
            sys.argv.append(folder)
            
    print(sys.argv[1:])
    
    labelme_folder = sys.argv[1]
    converter(labelme_folder)
    print("Succeed")
