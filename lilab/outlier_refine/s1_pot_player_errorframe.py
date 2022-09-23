# python -m lilab.outlier_refine.s1_pot_player_errorframe /A/B/C
# %%
from PIL import Image, ImageTk
import base64
from aip import AipOcr
import io
import glob
import os.path as osp
import json
import tqdm
import pytesseract
import cv2
import numpy as np
import argparse
from lilab.mmocr.s1_ocr_readnum import MyOCR


dir = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/for_label_refine20220901/white/merge'

def dump_json_from_dir(dir):
    findext = lambda x: glob.glob(osp.join(dir, x))
    error_imgs = findext('*.png') + findext('*.jpg')
    error_imgs = [osp.basename(error_img) for error_img in error_imgs
                if '.mp4' in error_img]

    error_videos = [error_img.split('.mp4')[0]+'.mp4' for error_img in error_imgs]
    outdict = {error_video:[] for error_video in error_videos}
    faillist = []
    myOCR = MyOCR()
    for error_img, error_video in zip(tqdm.tqdm(error_imgs), error_videos):
        num = myOCR(osp.join(dir,error_img))
        if num:
            outdict[error_video].append(num)
        else:
            faillist.append(error_img)

    outjson = osp.join(dir, 'out.json')
    with open(outjson, 'w') as f:
        json.dump(outdict, f, indent=4)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    args = parser.parse_args()
    dump_json_from_dir(args.dir)
