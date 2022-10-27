# from lilab.mmdet_dev.canvas_reader import CanvasReader
import numpy as np
import ffmpegcv
import pickle
import pycocotools.mask as maskUtils
from lilab.mmlab_scripts.show_pkl_seg_video_fast import get_mask_colors
import torch
import torchvision
from multiprocessing import Pool
import mmcv
import cv2

class CanvasReader:
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
            mask_index = {0:(0,0), 1:(0,1), 2:(0,2), 
                            3:(1,0), 4:(1,1), 5:(1,2)}
        elif len(views_xywh)==10:
            vid = ffmpegcv.VideoReaderNV(video_path,
                                            gpu = gpu,
                                            pix_fmt='rgb24')
            mask_container_size = (4, 3, 800, 1280)
            mask_index = {0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2),
                      6:(2,0), 7:(2,1), 8:(2,2), 9:(3,0)}
        elif len(views_xywh)==9:
            vid = ffmpegcv.VideoReaderNV(video_path,
                                            gpu = gpu,
                                            pix_fmt='rgb24')
            mask_container_size = (3, 3, 800, 1280)
            mask_index = {0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2),
                      6:(2,0), 7:(2,1), 8:(2,2)}
        else:
            print('len(views_xywh) must be 6, 9 or 10. But get', len(views_xywh))
            raise NotImplementedError
        
        self.cuda = f'cuda:{gpu}'
        self.vid = vid
        self.views_xywh = views_xywh
        self.pkl_data = pkl_data
        self.nclass = 2
        with torch.cuda.device(self.cuda):
            self.mask_container = torch.zeros((self.nclass, *mask_container_size)).cuda().half()
            self.mask_index = mask_index
            self.dilate = dilate
            self.fun_resize = torchvision.transforms.Resize(self.mask_container.shape[-2:],
                                torchvision.transforms.InterpolationMode.NEAREST)
            self.mask_colors = torch.Tensor(get_mask_colors()).cuda().half()

    def read_canvas_mask_only(self):
        if self.iframe<0: return None
        segdata = self.pkl_data['dilate_segdata' if self.dilate else 'segdata']
        maskiclass_canvas_a = self.mask_container
        h, w = maskiclass_canvas_a.shape[-2:]
        for iview in range(len(self.views_xywh)):
            segdata_iview = segdata[iview][self.iframe]
            for iclass in range(self.nclass):
                mask = segdata_iview[1][iclass]
                mask = maskUtils.decode(mask)[:,:,0] if mask else np.zeros((h, w), dtype=np.uint8)
                mask_cuda = torch.from_numpy(mask).cuda(self.cuda).half()
                if mask_cuda.shape[0] != h or mask_cuda.shape[1] != w:
                    mask_cuda = self.fun_resize(mask_cuda[None,...])[None]
                index = self.mask_index[iview]
                maskiclass_canvas_a[iclass][index[0]][index[1]] = mask_cuda
        
        maskiclass_canvas_c = maskiclass_canvas_a.permute(0,1,3,2,4)
        shape_c = maskiclass_canvas_c.shape
        maskiclass_canvas_d = maskiclass_canvas_c.reshape(shape_c[0],
                                                        shape_c[1]*shape_c[2],
                                                        shape_c[3]*shape_c[4])
        
        maskiclass_canvas_e = maskiclass_canvas_d>0   # (nclass, h, w)
        return maskiclass_canvas_e

    def read_canvas_mask_img(self, img=None):
        if img is None:
            ret, img = self.read()
            if not ret: return None
        maskiclass_canvas = self.read_canvas_mask_only()
        frame_cuda = torch.from_numpy(img.copy()).cuda(self.cuda).half()
        for iclass in range(self.nclass):
            frame_cuda[maskiclass_canvas[iclass]] = frame_cuda[maskiclass_canvas[iclass]]/2 + self.mask_colors[iclass]/2
        frame_np = frame_cuda.type(torch.uint8).cpu().numpy()
        return frame_np
    
    def read_canvas_mask_img_out(self, img=None):
        if img is None:
            ret, img = self.read()
            if not ret: return None
        maskiclass_canvas = self.read_canvas_mask_only()
        frame_cuda = torch.from_numpy(img.copy()).cuda(self.cuda).half()
        frame_iclass = []
        for iclass in range(self.nclass):
            frame_with_mask_cuda = frame_cuda * maskiclass_canvas[iclass][:,:,None].half()
            frame_with_mask_np = frame_with_mask_cuda.type(torch.uint8).cpu().numpy()
            frame_iclass.append(frame_with_mask_np)
        return frame_iclass



class CanvasReaderCV:
    # set property 'iframe' is the self.vid.iframe
    @property
    def count(self):
        return len(self.vid)

    def __len__(self):
        return len(self.vid)

    def release(self):
        pass

    def __getitem__(self, iframe):
        canvasimg = self.vid[iframe]
        canvasimgrgb = cv2.cvtColor(canvasimg, cv2.COLOR_BGR2RGB)
        self.iframe = iframe
        return self.read_canvas_mask_img_out(canvasimgrgb)

    def __init__(self, video_path, segment_path=None, gpu=0, dilate=True):
        if segment_path is None:
            segment_path = video_path.replace('.mp4', '.segpkl')
        pkl_data = pickle.load(open(segment_path, 'rb'))
        views_xywh = pkl_data['views_xywh']
        if len(views_xywh)==6:

            mask_container_size = (2, 3, 600, 800)
            mask_index = {0:(0,0), 1:(0,1), 2:(0,2), 
                            3:(1,0), 4:(1,1), 5:(1,2)}
        elif len(views_xywh)==10:
            mask_container_size = (4, 3, 800, 1280)
            mask_index = {0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2),
                      6:(2,0), 7:(2,1), 8:(2,2), 9:(3,0)}
        elif len(views_xywh)==9:
            mask_container_size = (3, 3, 800, 1280)
            mask_index = {0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2),
                      6:(2,0), 7:(2,1), 8:(2,2)}
        else:
            raise NotImplementedError
        
        self.cuda = f'cuda:{gpu}'
        self.vid = mmcv.VideoReader(video_path)
        self.views_xywh = views_xywh
        self.pkl_data = pkl_data
        self.nclass = 2
        self.iframe = -1
        with torch.cuda.device(self.cuda):
            self.mask_container = torch.zeros((self.nclass, *mask_container_size)).cuda().half()
            self.mask_index = mask_index
            self.dilate = dilate
            self.fun_resize = torchvision.transforms.Resize(self.mask_container.shape[-2:],
                                torchvision.transforms.InterpolationMode.NEAREST)
            self.mask_colors = torch.Tensor(get_mask_colors()).cuda().half()

    def read_canvas_mask_only(self):
        if self.iframe<0: return None
        segdata = self.pkl_data['dilate_segdata' if self.dilate else 'segdata']
        maskiclass_canvas_a = self.mask_container
        h, w = maskiclass_canvas_a.shape[-2:]
        for iview in range(len(self.views_xywh)):
            segdata_iview = segdata[iview][self.iframe]
            for iclass in range(self.nclass):
                mask = segdata_iview[1][iclass]
                mask = maskUtils.decode(mask)[:,:,0] if mask else np.zeros((h, w), dtype=np.uint8)
                mask_cuda = torch.from_numpy(mask).cuda(self.cuda).half()
                if mask_cuda.shape[0] != h or mask_cuda.shape[1] != w:
                    mask_cuda = self.fun_resize(mask_cuda[None,...])[None]
                index = self.mask_index[iview]
                maskiclass_canvas_a[iclass][index[0]][index[1]] = mask_cuda
        
        maskiclass_canvas_c = maskiclass_canvas_a.permute(0,1,3,2,4)
        shape_c = maskiclass_canvas_c.shape
        maskiclass_canvas_d = maskiclass_canvas_c.reshape(shape_c[0],
                                                        shape_c[1]*shape_c[2],
                                                        shape_c[3]*shape_c[4])
        
        maskiclass_canvas_e = maskiclass_canvas_d>0   # (nclass, h, w)
        return maskiclass_canvas_e

    def read_canvas_mask_img(self, img):
        maskiclass_canvas = self.read_canvas_mask_only()
        frame_cuda = torch.from_numpy(img.copy()).cuda(self.cuda).half()
        for iclass in range(self.nclass):
            frame_cuda[maskiclass_canvas[iclass]] = frame_cuda[maskiclass_canvas[iclass]]/2 + self.mask_colors[iclass]/2
        frame_np = frame_cuda.type(torch.uint8).cpu().numpy()
        return frame_np
    
    def read_canvas_mask_img_out(self, img):
        maskiclass_canvas = self.read_canvas_mask_only()
        frame_cuda = torch.from_numpy(img.copy()).cuda(self.cuda).half()
        frame_iclass = []
        for iclass in range(self.nclass):
            frame_with_mask_cuda = frame_cuda * maskiclass_canvas[iclass][:,:,None].half()
            frame_with_mask_np = frame_with_mask_cuda.type(torch.uint8).cpu().numpy()
            frame_iclass.append(frame_with_mask_np)
        return frame_iclass


class CanvasReaderThumbnail(CanvasReader):
    def __init__(self, video_path, segment_path=None, gpu=0, dilate=True):
        super().__init__(video_path, segment_path, gpu, dilate)
        self.vid.release()
        if len(self.views_xywh)==6:
            vid = ffmpegcv.VideoReaderNV(video_path,
                                            gpu = gpu,
                                            pix_fmt='rgb24',
                                            crop_xywh=[0,0,800*3,600*2],
                                            resize=(532*3,400*2),
                                            resize_keepratio=False)
            mask_container_size = (2, 3, 400, 532)
            scale_wh = (800/532, 600/400)
        elif len(self.views_xywh)==10:
            vid = ffmpegcv.VideoReaderNV(video_path,
                                            gpu = gpu,
                                            pix_fmt='rgb24',
                                            crop_xywh=[0,0,1280*3,800*4],
                                            resize=(640*3,400*4),
                                            resize_keepratio=False)
            mask_container_size = (4, 3, 400, 640)
            scale_wh = (1280/640, 800/400)
        elif len(self.views_xywh)==9:
            vid = ffmpegcv.VideoReaderNV(video_path,
                                            gpu = gpu,
                                            pix_fmt='rgb24',
                                            crop_xywh=[0,0,1280*3,800*3],
                                            resize=(640*3,400*3),
                                            resize_keepratio=False)
            mask_container_size = (3, 3, 400, 640)
            scale_wh = (1280/640, 800/400)
        else:
            raise NotImplementedError
        self.scale_wh = scale_wh
        self.vid = vid
        self.mask_container = torch.zeros((self.nclass, *mask_container_size), device=self.cuda, dtype=torch.half)
        self.fun_resize = torchvision.transforms.Resize(self.mask_container.shape[-2:],
                            torchvision.transforms.InterpolationMode.NEAREST)
