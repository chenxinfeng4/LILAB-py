# python -m lilab.mmlab_scripts.show_pkl_imagefolder ./data/mmdetection/test_pkl
# %% load packages
import mmcv
import numpy as np
import pycocotools._mask as mask_util
from tqdm import tqdm
import os
import argparse


# %% define the function to draw the mask and the bounding box
from lilab.mmlab_scripts.show_pkl_imagefolder import imshow_det_bboxes

# %% load pkl
'''
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

data[iframe][0=bbox][0|1=iclass][ianimal*XYXYP] = numpy.array
data[iframe][1=segm][0|1=iclass][ianimal]   = dict
'''
def pkl_2_images(pkl_data, pkl_datafilename):
    data = mmcv.load(pkl_data)
    datafilename = mmcv.load(pkl_datafilename)
    # get the folder name of the pkl file
    folder = os.path.join(os.path.dirname(pkl_datafilename), 'mask_labeded')
    os.makedirs(folder, exist_ok=True)
    
    class_names = ['rat', ]
    class_nicknames = ['rat', ]

    for label, imgfilename in zip(tqdm(data), datafilename):
        bboxes, segms, labels = [], [], []
        for iclass, _ in enumerate(class_names):
            if len(label[0][iclass]):
                bboxes.append(label[0][iclass]) #append numpy.array
                segms.extend(label[1][iclass])  #extend list
                labels.extend([iclass]*len(label[1][iclass]))

        img = mmcv.imread(imgfilename)
        bboxes = np.concatenate(bboxes)
        labels = np.array(labels, dtype='int')
        masks  = mask_util.decode(segms).transpose((2,0,1))
        img    = imshow_det_bboxes(img, bboxes,labels,masks,class_nicknames, 
                                show=False, bbox_color='white', show_bbox=False,
                                instance_mode=True)
        # save the image
        mmcv.imwrite(img, os.path.join(folder, os.path.basename(imgfilename)))
        


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
    pkl_2_images(pkl_data, pkl_datafilename)
    