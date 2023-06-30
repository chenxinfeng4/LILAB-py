# python -m lilab.mmlab_scripts.show_pkl_seg_video ./data/mmdetection/test_pkl
# %% load packages
import mmcv
import numpy as np
import pycocotools._mask as mask_util
import pickle
import tqdm 
import os
import argparse
from multiprocessing import Pool
from .segout_pkl_video import get_masks, dilate_masks, focus_image_by_masks
import shutil
from lilab.cvutils.auto_find_subfolderimages import find_subfolderfiles


"""create a class that generate the image folder """
class SegImagefolder:
    def ImageCapture(self, dataseg, datafilename):
        self.dataseg = dataseg
        self.datafilename = datafilename
        assert len(self.dataseg) == len(self.datafilename), 'data and datafilename should have the same length'

        # generate the output folder 
        self.__outfolder = os.path.join(os.path.dirname(self.datafilename[0]), 'seg_out')
        os.makedirs(self.__outfolder, exist_ok=True)

        # check the number of classes should be 1 or 2
        self.nclass = len(self.dataseg[0][0])
        assert self.nclass in [1, 2], 'the number of classes should be 1 or 2'
        self.ratclasses = ['rat'] if self.nclass==1 else ['rat_black', 'rat_white']
        
        return self

    def __len__(self):
        return len(self.dataseg)

    def __getitem__(self, idx):
        label, imgfilename = self.dataseg[idx], self.datafilename[idx]
        img = mmcv.imread(imgfilename)
        # get the width and height of the image
        self.height, self.width = img.shape[:2]
        # get the masks
        masks = get_masks(self.dataseg, idx, self.width, self.height)
        # dilate the masks
        masks = dilate_masks(masks, kernel_size=31)
        # focus the image by the masks
        focus_imgs = focus_image_by_masks(img, masks)
        return focus_imgs, imgfilename

    def write(self, focus_imgs, imgfilename):
        # get the nake name of the imgfilename
        imgNakename,extName = os.path.splitext(os.path.basename(imgfilename))
        # write the images into the folder
        for focus_img, ratclass in zip(focus_imgs, self.ratclasses):
            if self.nclass==1:
                outfilename = os.path.join(self.__outfolder, '{}{}'.format(imgNakename, extName))
            else:
                outfilename = os.path.join(self.__outfolder, '{}_{}{}'.format(imgNakename, ratclass, extName))
            mmcv.imwrite(focus_img[:,:,1], outfilename)

    def release(self):
        pass

    def delfolder(self):
        if os.path.exists(self.__outfolder) and os.path.isdir(self.__outfolder):
            shutil.rmtree(self.__outfolder)


def main(dataseg, datafilename):
    segvideo = SegImagefolder().ImageCapture(dataseg, datafilename)
    # try:
    for i in tqdm.tqdm(range(len(segvideo))):
        focus_imgs, imgfilename = segvideo[i]
        segvideo.write(focus_imgs, imgfilename)
    segvideo.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pkl to video')
    parser.add_argument(
        'cocopkl_file',
        help='pkl file',
        type=str)
    args = parser.parse_args()

    # check the existence of the pkl
    if not os.path.isfile(args.cocopkl_file):
        raise FileNotFoundError(f'{args.cocopkl_file} not found')

    data = pickle.load(open(args.cocopkl_file, 'rb'))
    dataseg = data['segdata']
    datafilename = data['imgfiles']
    datafilename = find_subfolderfiles(args.cocopkl_file, datafilename)
    # process the pkl data to images
    main(dataseg, datafilename)