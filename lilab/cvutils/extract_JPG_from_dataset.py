# python -m lilab.cvutils.extract_JPG_from_dataset
'''
Author: your name
Date: 2021-10-06 13:15:24
LastEditTime: 2021-10-06 14:21:52
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \BCNet\extract_JPG_from_dataset.py
'''

import json
import shutil
import os
import os.path as osp
import argparse
import glob

def main(cocofile):
    with open(cocofile, 'r') as f:
        data = json.load(f)
    imgs = data['images']
    files = [img['file_name'] for img in imgs]

    file0, file1 = files[0], files[-1]
    jsonfolder, jsonname = osp.split(cocofile)
    if osp.isfile(osp.join(jsonfolder, file0)):
        img_prefix = jsonfolder
    else:
        currentlist = [osp.join(jsonfolder, d) for d in os.listdir(jsonfolder)]
        currentfolder = [d for d in currentlist if osp.isdir(d) and 'extract' not in d]
        for folder in currentfolder:
            if osp.isfile(osp.join(folder, file0)) and osp.isfile(osp.join(folder, file1)):
                img_prefix = folder
                break
        else:
            img_prefix = input("Please give [Image Prefix] >> ")

    assert osp.isfile(osp.join(img_prefix, file0))
    outfoldername = osp.join(jsonfolder, 'extract_'+ osp.splitext(jsonname)[0])
    os.makedirs(outfoldername, exist_ok=True)

    # copy file to output folder
    for file in files:
        file_glob = glob.glob(osp.join(img_prefix, osp.splitext(file)[0]+'.*'))
        for f in file_glob:
            shutil.copy(f, outfoldername)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cocofile', type=str, help='path to coco json file')
    args = parser.parse_args()
    assert osp.isfile(args.cocofile)
    main(args.cocofile)
    print('Done!')