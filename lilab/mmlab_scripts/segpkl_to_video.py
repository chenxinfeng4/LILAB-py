import os
import numpy as np
import pycocotools._mask as mask_util
import cv2
import mmcv
import tqdm
import argparse
import glob
from multiprocessing import Pool

class_names = ['rat_black', 'rat_white']
class_nicknames = ['black', 'white']

def get_masks(pkl_data, idx, width, height):
    label = pkl_data[idx]    
    masks = []
    for iclass in range(len(label[1])):
        segms = label[1][iclass]
        if len(segms) == 0:
            mask = np.zeros((height, width), dtype=np.bool)
        elif len(segms) == 1:
            mask = mask_util.decode(segms).transpose((2,0,1)).astype(bool)[0]
        else:
            masks_mult = mask_util.decode(segms).transpose((2,0,1))
            areas = np.sum(masks_mult, axis=(1,2))
            bboxes = label[0][iclass]
            pvals  = bboxes[:,-1]
            rankvals = areas*pvals
            # sort the masks_mult by the rankvals
            sort_idx = np.argsort(rankvals)
            masks_mult = masks_mult[sort_idx]
            mask = masks_mult[0].astype(bool)
        masks.append(mask)
    return masks

"""dilate the mask by 50 pixels"""
def dilate_masks(masks, kernel_size=50):
    out_masks = []
    for mask in masks:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask.astype(np.uint8).copy(), kernel)
        out_masks.append(mask.astype(bool))
    return out_masks



"""create a class that generate the video """
class SegVideo:
    ratclasses = ['rat_black', 'rat_white']
    def __init__(self) -> None:
        pass

    def VideoCapture(self, pkl):
        self.pkl = pkl
        video_path = pkl.replace('.pkl', '.mp4')
        # check the pkl file and video file existed
        assert os.path.exists(self.pkl), '{} not found'.format(self.pkl)
        assert os.path.exists(video_path), '{} not found'.format(video_path)

        data = mmcv.load(self.pkl)
        v = mmcv.VideoReader(video_path)
        
        self.data, self.v = data, v
        self.n_frames = len(data)
        assert len(v) == len(data)

        # get the frame rate, width and heigth of the video
        self.width, self.height = v.width, v.height

        """get the parent name of the folder"""
        self.__infile =  video_path
        self.__outfile = [video_path.replace('.mp4','_blackmask.avi'), video_path.replace('.mp4','_whitemask.avi')]
        self.__outfilefinal = [video_path.replace('.mp4','_black.avi'), video_path.replace('.mp4','_white.avi')]
        self.out1 = cv2.VideoWriter(self.__outfile[0], cv2.VideoWriter_fourcc(*'MJPG'), v.fps, (v.width, v.height))
        self.out2 = cv2.VideoWriter(self.__outfile[1], cv2.VideoWriter_fourcc(*'MJPG'), v.fps, (v.width, v.height))
        return self

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx):
        #img = self.v[idx]
        masks = get_masks(self.data, idx, self.width, self.height)
        masks = dilate_masks(masks, kernel_size=30)
        mask = masks[0]
        # convert the mask to to rgb channel
        focus_imgs = [cv2.cvtColor(mask.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR) for mask in masks]
        #focus_imgs = focus_image_by_masks(img, masks)
        return focus_imgs

    def write(self, focus_imgs):
        self.out1.write(focus_imgs[0])
        self.out2.write(focus_imgs[1])

    def release(self):
        self.out1.release()
        self.out2.release()

    def mergemask(self):
        for outfilemask, outfilefinal in zip(self.__outfile, self.__outfilefinal):
            cmd = f'ffmpeg -i "{outfilemask}" -i "{self.__infile}"  -filter_complex "[0:v]colorkey=white:0.1[ckout];[1:v][ckout]overlay=0:0" -y "{outfilefinal}"'
            os.system(cmd)

    def delvideos(self):
        self.release()
        """if exist the __outfile, then remove it"""
        for outfile in self.__outfile:
            if os.path.exists(outfile):
                os.remove(outfile)

def main(pkl):
    segvideo = SegVideo().VideoCapture(pkl)
    try:
        for i in tqdm.tqdm(range(len(segvideo))):
            focus_imgs = segvideo[i]
            segvideo.write(focus_imgs)
        segvideo.release()
        segvideo.mergemask()
        print('Done in video <{}>'.format(pkl))
    except :
        # print a error message and delete the videos
        print('ERROR in video <{}>'.format(pkl))
        segvideo.delvideos()


# %% __main__
# parse the arguments.
# inputs: pkl_folder
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pkl to video')
    parser.add_argument(
        'pkl_folder',
        help='pkl folder or pkl file',
        type=str)
    args = parser.parse_args()

    # check the existence of the pkl folder
    if os.path.isdir(args.pkl_folder):
        pkl_files = glob.glob(os.path.join(args.pkl_folder, '*.pkl'))
    elif os.path.isfile(args.pkl_folder):
        pkl_files = [args.pkl_folder]
    else:
        raise FileNotFoundError(f'{args.pkl_folder} not found')

    with Pool(processes=6) as pool:
        pool.map(main, pkl_files)
