# python -m lilab.mmlab_scripts.show_pkl_seg_video ./data/mmdetection/test_pkl
# %% load packages
import mmcv
import numpy as np
import pycocotools._mask as mask_util
import tqdm 
import os
import argparse
from multiprocessing import Pool
from .segout_pkl_video import get_masks, dilate_masks, focus_image_by_masks
import shutil



"""create a class that generate the image folder """
class SegImagefolder:
    def ImageCapture(self, pkl_data, pkl_datafilename):
        self.pkl_data = pkl_data
        self.pkl_datafilename = pkl_datafilename
        # check the pkl file and video file existed
        assert os.path.exists(self.pkl_data), '{} not found'.format(self.pkl_data)
        assert os.path.exists(self.pkl_datafilename), '{} not found'.format(self.pkl_datafilename)

        self.data = mmcv.load(self.pkl_data)
        self.datafilename = mmcv.load(self.pkl_datafilename)
        assert len(self.data) == len(self.datafilename), 'data and datafilename should have the same length'

        # generate the output folder 
        self.__outfolder = os.path.join(os.path.dirname(self.pkl_data), 'seg_out')
        os.makedirs(self.__outfolder, exist_ok=True)

        # check the number of classes should be 1 or 2
        self.nclass = len(self.pkl_data[0][0])
        assert self.nclass in [1, 2], 'the number of classes should be 1 or 2'
        self.ratclasses = ['rat'] if self.nclass==1 else ['rat_black', 'rat_white']
        
        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, imgfilename = self.data[idx], self.datafilename[idx]
        img = mmcv.imread(imgfilename)
        # get the width and height of the image
        self.height, self.width = img.shape[:2]
        # get the masks
        masks = get_masks(self.data, idx, self.width, self.height)
        # dilate the masks
        masks = dilate_masks(masks, kernel_size=30)
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
            mmcv.imwrite(focus_img, outfilename)

    def release(self):
        pass

    def delfolder(self):
        if os.path.exists(self.__outfolder) and os.path.isdir(self.__outfolder):
            shutil.rmtree(self.__outfolder)

def main(pkl_data, pkl_datafilename):
    segvideo = SegImagefolder().ImageCapture(pkl_data, pkl_datafilename)
    # try:
    for i in tqdm.tqdm(range(len(segvideo))):
        focus_imgs, imgfilename = segvideo[i]
        segvideo.write(focus_imgs, imgfilename)
    segvideo.release()
    print('Done in folder <{}>'.format(pkl_data))
    # except :
    #     # print a error message and delete the videos
    #     print('ERROR in folder <{}>'.format(pkl_data))
    #     segvideo.delfolder()


# %% __main__
# parse the arguments.
# inputs: pkl_folder
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pkl to video')
    parser.add_argument(
        'pkl_folder',
        help='pkl folder',
        type=str)
    args = parser.parse_args()

    # check the existence of the pkl folder
    if os.path.isdir(args.pkl_folder):
        pkl_data = os.path.join(args.pkl_folder, 'data.pkl')
        pkl_datafilename = os.path.join(args.pkl_folder, 'data_filename.pkl')
    else:
        raise FileNotFoundError(f'{args.pkl_folder} not found')

    # process the pkl data to images
    main(pkl_data, pkl_datafilename)