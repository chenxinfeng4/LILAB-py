# python -m lilab.mmpose.s1_labelme_2_cocokeypoint 
# %%
import json
import os
import os.path as osp
import glob
# %%
keypoint_cls = ['Nose', 'EarL', 'EarR', 'Neck', 'Back', 'Tail',
                'ForeShoulderL', 'ForePowL', 'ForeShoulderR',
                'ForePowR', 'BackShoulderL', 'BackPowL',
                'BackShoulderR', 'BackPowR']
keypoint_dict = {kpt: i for i, kpt in enumerate(keypoint_cls)}

# %%
labelme_dir = '/home/liying_lab/chenxinfeng/DATA/mmpose/data/rat_keypoints/bw_rat_kpt_1280x800_0425/'
labelme_jsons = glob.glob(osp.join(labelme_dir, '*.json'))

# %%
coco_data = {
    "info": {
    "description": "Rat Dataset",
    "version": 1.0,
    "year": 2020
  },
  "licenses": "",
  "categories": [
    {
      "id": 1,
      "keypoints": keypoint_cls,
      "name": "rat",
      "supercategory": "rat"
    }
  ],
  "images": [],
  "annotations": [],
}
# %%
imageid = 0
instanceid = 0

def append_img(file_name, height, width):
    global imageid
    imageid += 1
    coco_data['images'].append({
        "id": imageid,
        "file_name": file_name,
        "height": height,
        "width": width,
    })
    return imageid


def append_anno(image_id, height, width, keypoint_list):
    global instanceid
    instanceid += 1
    area = int(height * width)
    bbox = [0, 0, width, height]
    category_id = 1
    iscrowd = 0
    keypoints = keypoint_list
    num_keypoints = sum(1 for vis in keypoints[2::3] if vis)
    coco_data['annotations'].append({
        "id": instanceid,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": iscrowd,
        "keypoints": keypoints,
        "num_keypoints": num_keypoints
    })


# %%
for labelme_json in labelme_jsons:
    labelme_json_data = json.load(open(labelme_json, 'r'))

    # 1. get image info
    if 'imageHeight' not in labelme_json_data: continue
    height, width = labelme_json_data['imageHeight'], labelme_json_data['imageWidth']
    file_name = labelme_json_data['imagePath'].split('/')[-1]
    image_id = append_img(file_name, height, width)

    # 2. get keypoint info
    shapes = labelme_json_data['shapes']
    shapes.sort(key=lambda a:a['label'])
    labels = [shape['label'] for shape in shapes]
    rat_labels = [label.split('_')[0] for label in labels]
    kpt_labels = labels
    kpt_positions = [shape['points'][0] for shape in shapes]
    
    # 3. keypoint info to annotations
    kpt_list = [[0,0,0] for _ in range(len(keypoint_cls))]
    for kpt_label, kpt_position in zip(kpt_labels, kpt_positions):
        kpt_list[keypoint_dict[kpt_label]] = [int(kpt_position[0]), int(kpt_position[1]), 2]
    keypoint_list = sum(kpt_list, [])
    append_anno(image_id, height, width, keypoint_list)

# 4. save coco json
out_coco_json = osp.join(labelme_dir, osp.basename(osp.abspath(labelme_dir)) + '_tranval.json')
json.dump(coco_data, open(out_coco_json, 'w'), indent=2)

# %%
