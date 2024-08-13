# from lilab.cvutils_new.canvas_reader_pannel import CanvasReaderPannelMask
# %%

import ffmpegcv
from lilab.cameras_setup import get_view_xywh_wrapper
import pickle
import pycocotools.mask as maskUtils
import numpy as np
import cv2


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
                                            pix_fmt='gray',
                                            crop_xywh=[0,0,800*3,600*2])
        elif len(views_xywh) in (4, 9, 10):
            vid = ffmpegcv.VideoReaderNV(video_path,
                                            gpu = gpu,
                                            pix_fmt='gray')
        elif len(views_xywh)==5:
            vid = ffmpegcv.VideoReaderNV(video_path,
                                            gpu = gpu,
                                            pix_fmt='gray')
        else:
            raise NotImplementedError
        
        self.vid = vid
        self.views_xywh = views_xywh
        self.nview = len(views_xywh)
        self.pkl_data = pkl_data
        self.nclass = 1

    def read(self):
        #ret, frame = self.vid.read_gray()
        ret, frame = self.vid.read()
        if not ret: return ret,[]
        # frame = frame.copy()
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
        return self.nframe

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
            vid = ffmpegcv.noblock(ffmpegcv.VideoReaderNV,
                                               video_path,
                                            gpu = gpu,
                                            pix_fmt='gray',
                                            crop_xywh=[0,0,800*3,600*2])
            h_w = (600, 800)
        elif len(views_xywh)==9:
            vid = ffmpegcv.noblock(ffmpegcv.VideoReaderNV,
                                                video_path,
                                                gpu = gpu,
                                                pix_fmt='gray')
            h_w = (800, 1280)
        elif len(views_xywh)==10:
            vid = ffmpegcv.noblock(ffmpegcv.VideoReaderNV,
                                               video_path,
                                            gpu = gpu,
                                            pix_fmt='gray')
            h_w = (800, 1280)
        else:
            raise NotImplementedError
        
        self.vid = vid
        self.views_xywh = views_xywh
        self.nview = len(views_xywh)
        self.pkl_data = pkl_data
        self.nclass = len(pkl_data['segdata'][0][0][1])
        #self.iframe = self.iframe
        self.nframe = len(pkl_data['segdata'][0])
        self.h_w = h_w
        self.dilate = dilate
        self.dataset = MaskDataset(pkl_data,self.dilate,self.h_w,self.nclass,self.nview)
        self.out_numpy_shape = [*self.dataset.out_numpy_shape, 1]

    def read_canvas_mask_img_out_old(self):
        ret, imgpannels = self.read()
        assert ret
        maskiclass_canvas_torch = next(self.dataloader_iter)
        maskiclass_canvas_a = maskiclass_canvas_torch.numpy()
        mask_out_imgs = [[None for i in range(self.nview)] for j in range(self.nclass)]
        for iview in range(self.nview):
            for iclass in range(self.nclass):
                mask_out_imgs[iclass][iview] = imgpannels[iview] * maskiclass_canvas_a[iclass][iview][:,:,None]
        return mask_out_imgs
    
    def read_canvas_mask_img_out(self):
        ret, imgpannels = self.read()
        assert ret
        maskiclass_canvas_a = self.dataset[self.iframe]
        mask_out_imgs = np.array(imgpannels)[None] * maskiclass_canvas_a[...,None]
        return mask_out_imgs

    def read(self):
        ret, frame = self.vid.read()
        if not ret: return []
        imgpannels = [frame[y:y+h, x:x+w] for x, y, w, h in self.views_xywh]
        return ret, imgpannels

    def read_mask_img(self):
        pass

    def read_mask_only(self):
        if self.iframe<0: return None
        maskiclass_canvas_a = self.dataset[self.iframe]        
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


class MaskDataset:
    def __init__(self, pkldata, dilate=True, h_w=(800,1280), nclass=2, nview=10):
        self.pkl_data = pkldata
        self.dilate = dilate
        self.h_w = h_w
        self.nclass = nclass
        self.nview  = nview
        self.segdata = self.pkl_data['dilate_segdata' if self.dilate else 'segdata']
        self.maskEmpty = maskUtils.encode(np.zeros(h_w,dtype='uint8',order='F'))
        self.out_numpy_shape = (nclass, nview, h_w[0], h_w[1])

    def __len__(self):
        return len(self.pkl_data['segdata'][0])

    def __getitem__(self, index) :
        self.iframe = index
        return self.read_mask_only()

    def read_mask_only_old(self):
        if self.iframe<0: return None
        segdata = self.segdata
        maskiclass_canvas_a = [[None for _ in range(self.nview)] for _ in range(self.nclass)]
        
        maskenc_list = []
        for iclass in range(self.nclass):
            for iview in range(self.nview):
                segdata_iview = segdata[iview][self.iframe]
                mask = segdata_iview[1][iclass]
                if mask:
                    maskenc_list.extend(mask)
                else:
                    maskenc_list.extend(self.maskEmpty)
                
        mask = maskUtils.decode(maskenc_list)
        maskiclass_canvas=np.reshape(mask, (mask.shape[0], mask.shape[1],self.nview,self.nclass), order='F') 
        maskiclass_canvas_a = np.transpose(maskiclass_canvas, [3,2,0,1]) #nclass, nview, H, W
        maskiclass_canvas_a = torch.from_numpy(maskiclass_canvas_a)
        return maskiclass_canvas_a

    def read_mask_only(self):
        if self.iframe<0: return None
        segdata = self.segdata
        maskenc_list = []
        for iclass in range(self.nclass):
            for iview in range(self.nview):
                segdata_iview = segdata[iview][self.iframe]
                mask = segdata_iview[1][iclass]
                if mask:
                    maskenc_list.extend(mask)
                else:
                    maskenc_list.extend(self.maskEmpty)

        mask = maskUtils.decode_orderc_channelfirst(maskenc_list)
        maskiclass_canvas=np.reshape(mask, (self.nclass, self.nview, mask.shape[1], mask.shape[2])) #nclass, nview, H, W
        return maskiclass_canvas
        