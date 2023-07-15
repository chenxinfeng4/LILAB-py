# conda activate mmdet
# python -m lilab.outlier_refine.s1_pot_player_errorframe /A/B/C
# %%
import glob
import os.path as osp
import json
import tqdm
import argparse
from lilab.mmocr.s1_ocr_readnum import MyOCR


dir = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/for_label_refine20220901/white/merge'

def dump_json_from_dir(myOCR, dir):
    findext = lambda x: glob.glob(osp.join(dir, x))
    error_imgs = findext('*.png') + findext('*.jpg')
    error_imgs = [osp.basename(error_img) for error_img in error_imgs
                if '.mp4' in error_img]

    error_videos = [error_img.split('.mp4')[0]+'.mp4' for error_img in error_imgs]
    outdict = {error_video:[] for error_video in error_videos}
    faillist = []
    
    for error_img, error_video in zip(tqdm.tqdm(error_imgs), error_videos):
        num = myOCR(osp.join(dir,error_img))
        if num:
            outdict[error_video].append(int(num))
        else:
            faillist.append(error_img)

    outjson = osp.join(dir, 'out.json')
    with open(outjson, 'w') as f:
        json.dump(outdict, f, indent=4)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', type=str, nargs='+')
    args = parser.parse_args()
    assert len(args.dirs)
    myOCR = MyOCR()
    for dir in args.dirs:
        dump_json_from_dir(myOCR, dir)
