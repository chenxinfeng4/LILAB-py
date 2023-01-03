# python -m lilab.outlier_refine.s0_pot_player_errorframe_crop /A/B/C/
# %%
from PIL import Image, ImageTk
import base64
import io
import glob
import os.path as osp
import json
import tqdm
import os
import argparse
# imgfile = '2022-04-26_15-06-02_bwt_wwt_7.mp4_20220509_155402.017.png'
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

dir = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/for_label_refine20220901/white'
crop_xyxy = [20, 2, 250, 70]
# crop_xyxy = [0, 0, 150, 70]
#%% 
def get_num(imgfile, outdir):
    img = Image.open(imgfile)
    img_num = img.crop(crop_xyxy)
    outimgfile = osp.join(outdir, osp.basename(imgfile))
    img_num.save(outimgfile, format="PNG")
    

def crop(dir):
    findext = lambda x: glob.glob(osp.join(dir, x))
    error_imgs = findext('*.png') + findext('*.jpg')
    error_imgs = [osp.basename(error_img) for error_img in error_imgs
                if '.mp4' in error_img]

    error_videos = [error_img.split('.mp4')[0]+'.mp4' for error_img in error_imgs]
    index_processed = [False for _ in range(len(error_imgs))]
    outdict = {error_video:[] for error_video in error_videos}
    for i, (error_img, error_video) in enumerate(zip(tqdm.tqdm(error_imgs), error_videos)):
        outdir = osp.join(dir, 'out_'+error_video)
        os.makedirs(outdir, exist_ok=True)
        get_num(osp.join(dir,error_img), outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='dir')
    args = parser.parse_args()
    crop(args.dir)