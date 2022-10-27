# from lilab.cvutils_new.canvas_reader_pannel import CanvasReaderPannelMask
# %%

import ffmpegcv
from lilab.cameras_setup import get_view_xywh_wrapper
import pickle
import pycocotools.mask as maskUtils
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2


class CanvasReaderPannel(object):
    def __init__(self, video_path, gpu=0):
        self.vid = ffmpegcv.VideoCaptureNV(video_path, gpu=gpu)
        if self.vid.size==(3840, 3200):
            self.view_xywh = get_view_xywh_wrapper(10)
        elif self.vid.size==(2560, 1440):
            self.view_xywh = get_view_xywh_wrapper(6)
        elif self.vid.size==(1280*2, 800*2):
            self.view_xywh = get_view_xywh_wrapper(4)
        else:
            raise ValueError('unknown video size')
        
        self.release = self.vid.release
        self.__len__ = self.vid.__len__

    def read(self):
        ret, frame = self.vid.read()
        if not ret: return []
        outpannels = [frame[y:y+h, x:x+w] for x, y, w, h in self.view_xywh]
        return outpannels


class CanvasReaderPannel(object):
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

    def release(self):
        return self.vid.release()

    def __init__(self, video_path, segment_path=None, gpu=0, dilate=True):
        if segment_path is None:
            segment_path = video_path.replace('.mp4', '.segpkl')
        if isinstance(segment_path, dict):
            pkl_data = segment_path
        else:
            pkl_data = pickle.load(open(segment_path, 'rb'))
        views_xywh = pkl_data['views_xywh']
        if len(views_xywh)==6:
            vid = ffmpegcv.VideoReaderNV(video_path,
                                            gpu = gpu,
                                            pix_fmt='nv12',
                                            crop_xywh=[0,0,800*3,600*2])
        elif len(views_xywh) in (4, 9, 10):
            vid = ffmpegcv.VideoReaderNV(video_path,
                                            gpu = gpu,
                                            pix_fmt='nv12')
        else:
            raise NotImplementedError
        
        self.vid = vid
        self.views_xywh = views_xywh
        self.nview = len(views_xywh)
        self.pkl_data = pkl_data
        self.nclass = 1

    def read(self):
        ret, frame = self.vid.read_gray()
        if not ret: return []
        frame = frame.copy()
        imgpannels = [frame[y:y+h, x:x+w] for x, y, w, h in self.views_xywh]
        return ret, [imgpannels]


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
        if isinstance(segment_path, dict):
            pkl_data = segment_path
        else:
            pkl_data = pickle.load(open(segment_path, 'rb'))
        views_xywh = pkl_data['views_xywh']
        if len(views_xywh)==6:
            vid = ffmpegcv.VideoReaderNV(video_path,
                                            gpu = gpu,
                                            pix_fmt='nv12',
                                            crop_xywh=[0,0,800*3,600*2])
            h_w = (600, 800)
        elif len(views_xywh)==9:
            vid = ffmpegcv.VideoReaderNV(video_path,
                                            gpu = gpu,
                                            pix_fmt='nv12')
            h_w = (800, 1280)
        elif len(views_xywh)==10:
            vid = ffmpegcv.VideoReaderNV(video_path,
                                            gpu = gpu,
                                            pix_fmt='nv12')
            h_w = (800, 1280)
        else:
            raise NotImplementedError
        
        self.vid = vid
        self.views_xywh = views_xywh
        self.nview = len(views_xywh)
        self.pkl_data = pkl_data
        self.nclass = 2
        self.h_w = h_w
        self.dilate = dilate
        dataset = MaskDataset(pkl_data,self.dilate,self.h_w,self.nclass,self.nview)
        dataloader = DataLoader(dataset, batch_size=None, num_workers=5)
        self.dataloader_iter = iter(dataloader)
        self.read_canvas_mask_img_out_mp = self.read_canvas_mask_img_out

    def read_canvas_mask_img_out(self):
        ret, imgpannels = self.read()
        assert ret
        maskiclass_canvas_torch = next(self.dataloader_iter)
        maskiclass_canvas_a = maskiclass_canvas_torch.numpy()
        mask_out_imgs = [[None for i in range(self.nview)] for j in range(self.nclass)]
        for iview in range(self.nview):
            for iclass in range(self.nclass):
                mask_out_imgs[iclass][iview] = imgpannels[iview] * maskiclass_canvas_a[iclass][iview][:,:,None]
        return mask_out_imgs

    def read(self):
        ret, frame = self.vid.read_gray()
        if not ret: return []
        frame = frame.copy()
        imgpannels = [frame[y:y+h, x:x+w] for x, y, w, h in self.views_xywh]
        return ret, imgpannels

    def read_mask_img(self):
        pass

    def read_mask_only(self):
        if self.iframe<0: return None
        segdata = self.pkl_data['dilate_segdata' if self.dilate else 'segdata']
        maskiclass_canvas_a = [[None for i in range(self.nview)] for j in range(self.nclass)]
        h, w = self.h_w
        for iview in range(self.nview):
            segdata_iview = segdata[iview][self.iframe]
            for iclass in range(self.nclass):
                mask = segdata_iview[1][iclass]
                mask = maskUtils.decode(mask)[:,:,0] if mask else np.zeros((h, w), dtype=np.uint8)
                if mask.shape[0] != h or mask.shape[1] != w:
                    # mask = cv2.resize(mask, [w,h],interpolation=cv2.INTER_NEAREST)
                    mask = cv2_resize(mask, [w,h],interpolation=cv2.INTER_NEAREST)
                    # mask = cv2.resize(mask, [w,h])
                maskiclass_canvas_a[iclass][iview] = mask
        
        return maskiclass_canvas_a

    def read_canvas_mask_img_out_single(self, imgpannels=None):
        if imgpannels is None:
            ret, imgpannels = self.read()
            if not ret: return None
        maskiclass_canvas = self.read_mask_only()
        mask_out_imgs = [[None for i in range(self.nview)] for j in range(self.nclass)]

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
                # Case 4, single cpu
                mask_out_imgs[iclass][iview] = imgpannels[iview] * maskiclass_canvas[iclass][iview][:,:,None]
                # Case 5, multple cpu
                # mask_out_imgs[iclass][iview] = (torch.from_numpy(imgpannels[iview]) * torch.from_numpy(maskiclass_canvas[iclass][iview])[:,:,None]).numpy()
        return mask_out_imgs

_map_dict = {}

def cv2_resize(img, size, interpolation=cv2.INTER_NEAREST):
    assert img.data.c_contiguous or img.data.f_contiguous
    assert interpolation == cv2.INTER_NEAREST
    assert len(img.shape)==2
    order = 'C' if img.data.c_contiguous else 'F'
    quary = (img.shape, tuple(size), order) #(h, w), (w2, h2), order
    if quary not in _map_dict:
        h1, w1 = img.shape[:2]
        w2, h2 = size
        h1_grid = np.arange(h1).astype('int64')
        w1_grid = np.arange(w1).astype('int64')
        h1_mesh, w1_mesh = np.meshgrid(h1_grid, w1_grid, indexing='ij')
        h2_mesh = cv2.resize(h1_mesh, (w2, h2), interpolation=cv2.INTER_NEAREST)
        w2_mesh = cv2.resize(w1_mesh, (w2, h2), interpolation=cv2.INTER_NEAREST)
        hw_coord2d = np.stack([h2_mesh, w2_mesh], axis=0)
        hw_coord_ravel = np.ravel_multi_index(hw_coord2d, dims=img.shape, order=order).astype(np.int64)
        _map_dict[quary] = hw_coord_ravel
    hw_coord_ravel = _map_dict[quary]
    img2 = img.ravel(order=order)[hw_coord_ravel]
    return img2


class MaskDataset(Dataset):
    def __init__(self, pkldata, dilate=True, h_w=(800,1280), nclass=2, nview=10):
        super().__init__()
        self.pkl_data = pkldata
        self.dilate = dilate
        self.h_w = h_w
        self.nclass = nclass
        self.nview  = nview
        self.segdata = self.pkl_data['dilate_segdata' if self.dilate else 'segdata']

    def __len__(self):
        return len(self.pkl_data['segdata'][0])

    def __getitem__(self, index) :
        self.iframe = index
        return self.read_mask_only()

    def read_mask_only(self):
        if self.iframe<0: return None
        segdata = self.segdata
        maskiclass_canvas_a = np.zeros((self.nclass, self.nview, *self.h_w), dtype=np.uint8)
        h, w = self.h_w
        for iview in range(self.nview):
            segdata_iview = segdata[iview][self.iframe]
            for iclass in range(self.nclass):
                mask = segdata_iview[1][iclass]
                mask = maskUtils.decode(mask)[:,:,0] if mask else np.zeros((h, w), dtype=np.uint8)
                if mask.shape[0] != h or mask.shape[1] != w:
                    mask = cv2_resize(mask, [w,h],interpolation=cv2.INTER_NEAREST)
                maskiclass_canvas_a[iclass][iview] = mask
        maskiclass_canvas_a = torch.from_numpy(maskiclass_canvas_a)
        return maskiclass_canvas_a
