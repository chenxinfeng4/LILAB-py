# %%
import ffmpegcv
from lilab.cameras_setup import get_view_xywh_wrapper
import pickle
import pycocotools.mask as maskUtils
import torch
import torchvision
import numpy as np
import cv2

class CanvasReaderPannel(object):
    def __init__(self, video_path, gpu=0):
        self.vid = ffmpegcv.VideoCaptureNV(video_path, gpu=gpu)
        if self.vid.size==(3840, 3200):
            self.view_xywh = get_view_xywh_wrapper(10)
        elif self.vid.size==(2560, 1440):
            self.view_xywh = get_view_xywh_wrapper(6)
        else:
            raise ValueError('unknown video size')
        
        self.release = self.vid.release
        self.__len__ = self.vid.__len__

    def read(self):
        ret, frame = self.vid.read()
        if not ret: return []
        outpannels = [frame[y:y+h, x:x+w] for x, y, w, h in self.view_xywh]
        return outpannels

class CanvasReaderPannelMask(object):
    # set property 'iframe' is the self.vid.iframe
    @property
    def iframe(self):
        return self.vid.iframe

    @property
    def count(self):
        return self.vid.count

    @property
    def fps(self):
        return self.vid.fps

    def __len__(self):
        return self.vid.count

    def read(self):
        return self.vid.read()

    def release(self):
        return self.vid.release()

    def __init__(self, video_path, segment_path=None, gpu=0, dilate=True):
        if segment_path is None:
            segment_path = video_path.replace('.mp4', '.segpkl')
        pkl_data = pickle.load(open(segment_path, 'rb'))
        views_xywh = pkl_data['views_xywh']
        if len(views_xywh)==6:
            vid = ffmpegcv.VideoReaderNV(video_path,
                                            gpu = gpu,
                                            pix_fmt='rgb24',
                                            crop_xywh=[0,0,800*3,600*2])
            mask_container_size = (2, 3, 600, 800)
        elif len(views_xywh)==10:
            vid = ffmpegcv.VideoReaderNV(video_path,
                                            gpu = gpu,
                                            pix_fmt='rgb24')
            mask_container_size = (4, 3, 800, 1280)
        else:
            raise NotImplementedError
        
        self.cuda = f'cuda:{gpu}'
        self.vid = vid
        self.views_xywh = views_xywh
        self.nview = len(views_xywh)
        self.pkl_data = pkl_data
        self.nclass = 2
        self.h_w = mask_container_size[-2:]
        self.dilate = dilate
        self.fun_resize = lambda img: cv2.resize(img, mask_container_size[-2:], 
                            interpolation=cv2.INTER_NEAREST)

    def read(self):
        ret, frame = self.vid.read()
        if not ret: return []
        outpannels = [frame[y:y+h, x:x+w] for x, y, w, h in self.views_xywh]
        return ret, outpannels

    def read_mask_img(self):
        pass

    def read_mask_only(self):
        if self.iframe<0: return None
        segdata = self.pkl_data['dilate_segdata' if self.dilate else 'segdata']
        maskiclass_canvas_a = [[None]*self.nclass]*self.nview
        h, w = self.h_w
        for iview in range(self.nview):
            segdata_iview = segdata[iview][self.iframe]
            for iclass in range(self.nclass):
                mask = segdata_iview[1][iclass]
                mask = maskUtils.decode(mask)[:,:,0] if mask else np.zeros((h, w), dtype=np.uint8)
                if mask.shape[0] != h or mask.shape[1] != w:
                    mask = self.fun_resize(mask)
                maskiclass_canvas_a[iview][iclass] = mask
        
        return maskiclass_canvas_a


    def read_canvas_mask_img_out(self, img=None):
        if img is None:
            ret, img = self.read()
            if not ret: return None
        maskiclass_canvas = self.read_mask_only()
        mask_out_imgs = [[None]*self.nclass]*self.nview

        for iview in range(self.nview):
            for iclass in range(self.nclass):
                # Case 1
                # mask = maskiclass_canvas[iview][iclass]
                # mask_out_imgs[iview][iclass] = img.copy()
                # mask_out_imgs[iview][iclass][mask==0] = 0
                # Case 2
                # mask_out_imgs[iview][iclass] = img[iview] * maskiclass_canvas[iview][iclass][:,:,None]
                # Case 3
                # mask_out_imgs[iview][iclass] = img[iview] * maskiclass_canvas[iview][iclass][:,:,None]
                # Case 4
                mask_out_imgs[iview][iclass] = img[iview][:,:,0] * maskiclass_canvas[iview][iclass]
        return mask_out_imgs
