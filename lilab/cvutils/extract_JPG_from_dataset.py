#!/usr/bin/python
'''
Author: your name
Date: 2021-10-06 13:15:24
LastEditTime: 2021-10-06 14:21:52
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \BCNet\extract_JPG_from_dataset.py
'''
#!pyinstaller -F extract_JPG_from_dataset.py -i logo_coco.ico
# python extract_JPG_from_dataset.py E:\cxf\BCNet\data\coco_rat_val.json
import json
import sys
import shutil
import os

if __name__ == '__main__':
    n = len(sys.argv)
    assert n==2
          
    print(sys.argv[1:])
    cocofile = sys.argv[1]
    with open(cocofile, 'r') as f:
        data = json.load(f)
    imgs = data['images']
    files = [img['file_name'] for img in imgs]

    file0 = files[0]
    jsonfolder, jsonname = os.path.split(cocofile)
    if os.path.isfile(os.path.join(jsonfolder, file0)):
        img_prefix = jsonfolder
    else:
        currentlist = [os.path.join(jsonfolder, d) for d in os.listdir(jsonfolder)]
        currentfolder = [d for d in currentlist if os.path.isdir(d) and 'extract' not in d]
        for folder in currentfolder:
            if os.path.isfile(os.path.join(folder, file0)):
                img_prefix = folder
                break
        else:
            img_prefix = input("Please give [Image Prefix] >> ")

    assert os.path.isfile(os.path.join(img_prefix, file0))
    outfoldername = os.path.join(jsonfolder, 'extract_'+ os.path.splitext(jsonname)[0])
    os.makedirs(outfoldername, exist_ok=True)

    # copy file to output folder
    for file in files:
        shutil.copy(os.path.join(img_prefix, file), outfoldername)
