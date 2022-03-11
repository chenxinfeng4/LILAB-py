# python -m lilab.mmlab_scripts.semanticimage_to_segpkl ./data/train/semanticimage 
# %%
import pycocotools.mask as maskUtils
import numpy as np
import cv2
import glob
import os.path as osp
import mmcv
import argparse

semanticimageFolder = '/home/liying_lab/chenxinfeng/DATA/mmsegmentation/data_seg/rats/test800x600_rater2_semanticLabel'


def convert(semanticimageFolder):
    imgfiles = glob.glob(osp.join(semanticimageFolder, '*.png')) +\
                glob.glob(osp.join(semanticimageFolder, '*.jpg'))
    outdir = osp.dirname(semanticimageFolder)
    datapkl = outdir + '/data.pkl'
    datafile = outdir + '/data_filename.pkl'
    nclass = 2
    outimgfiles = [osp.basename(osp.splitext(imgfile)[0]).replace('_labelTrainIds', '')+'.jpg'
                        for imgfile in imgfiles]
    mmcv.dump(outimgfiles, datafile)
    outdata = [[[[]] * nclass,[[]] * nclass] for _ in imgfiles]

    # %%
    for imgfile, frame_out in zip(imgfiles, outdata):
        img = cv2.imread(imgfile)[:,:,0]
        for iclass in range(nclass):
            mask = img == iclass+2
            if np.sum(mask) <= 4:
                # ignore this mask
                frame_out[0][iclass] = []
            else:
                x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
                pval = 1
                frame_out[0][iclass] = [np.array([x, y, x+w, y+h, pval])]

            frame_out[1][iclass] = maskUtils.encode(
                    np.array(mask[:,:,np.newaxis], order='F', dtype=np.uint8))

    mmcv.dump(outdata, datapkl)


# %% main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert semanticimage to segpkl')
    parser.add_argument('semanticimageFolder', default=semanticimageFolder, help='semanticimageFolder')
    args = parser.parse_args()
    convert(args.semanticimageFolder)