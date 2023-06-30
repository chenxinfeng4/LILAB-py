# python -m lilab.mmdet_dev.s1_mmdet_videos2pkl_trt  A.mp4 --pannels 9 [CONFIG] [CHECKPOINT]
# python -m lilab.mmdet_single.s1_mmdet_videos2pkl_single  A/B/C

import argparse
import glob
import itertools
import os
import os.path as osp
import pickle
import cv2

import ffmpegcv
import numpy as np
import torch
import tqdm
from mmdet2trt.apis import create_wrap_detector
from torchvision.transforms import functional as F
from ffmpegcv.video_info import get_info
import re
# from mmdet.models.roi_heads.mask_heads import FCNMaskHead
import lilab.mmdet_dev_multi.common_func as FCNMaskHead
from mmdet.core import bbox2result
from addict import Addict
import lilab.cvutils.map_multiprocess_cuda as mmap_cuda
from lilab.cameras_setup import get_view_xywh_wrapper
from mmdet.core import encode_mask_results
from mmdet.datasets.pipelines import Compose
from multiprocessing import Process, Queue
import multiprocess as mp
import itertools

# from lilab.mmpose_dev.a2_convert_mmpose2engine import findcheckpoint_trt

video_path = [
    f
    for f in glob.glob(
        "/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/test/*.mp4"
    )
    if f[-4] not in "0123456789"
]

config = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/mask_rcnn_r101_fpn_2x_coco_bwrat_816x512_cam9_oldrat.py'

def prefetch_img_metas(cfg, ori_wh):
    w, h = ori_wh
    cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = {"img": np.zeros((h, w, 3), dtype=np.uint8)}
    data = test_pipeline(data)
    img_metas = data["img_metas"][0].data
    return img_metas


def process_img(frame_resize, img_metas):
    if isinstance(frame_resize, np.ndarray):
        assert frame_resize.shape == img_metas["pad_shape"]
        frame_cuda = torch.from_numpy(frame_resize.copy()).cuda().float()
        frame_cuda = frame_cuda.permute(2, 0, 1)  # HWC to CHW
        frame_cuda = frame_cuda[None, :, :, :]  # NCHW
    else:
        frame_cuda = frame_resize.cuda().float()  # NHWC
        frame_cuda = frame_cuda.permute(0, 3, 1, 2)  # NCHW

    mean = torch.from_numpy(img_metas["img_norm_cfg"]["mean"]).cuda()
    std = torch.from_numpy(img_metas["img_norm_cfg"]["std"]).cuda()
    frame_cuda = F.normalize(frame_cuda, mean=mean, std=std, inplace=True)
    data = {"img": [frame_cuda], "img_metas": [[img_metas]]}
    return data

def findcheckpoint_trt(config, trtnake="latest.engine"):
    """Find the latest checkpoint of the model."""
    basedir = osp.dirname(config)
    basenakename = osp.splitext(osp.basename(config))[0]
    checkpoint = osp.join(basedir, "work_dirs", basenakename, trtnake)
    assert osp.isfile(checkpoint), "checkpoint not found: {}".format(checkpoint)
    return checkpoint

def refine_mask(masks0, masks1):
    if len(masks0) <=1 and len(masks1) <=1:
        pass
    elif len(masks0) == 0 or len(masks1) == 0:
        pass
    elif len(masks0) > 1 or len(masks1) > 1:
        focus_masks, parnter_masks = ((masks0, masks1), (masks1, masks0))
        for focus, parnter in zip(focus_masks, parnter_masks):
            if len(focus) == 1:
                continue
            # case 1: Focus main 接壤 Partner main, Focus sub 接壤 Partner sub。忽略
            partner_main = parnter[0].astype('uint8')
            focus_main = focus[0].astype('uint8')
            kernel = np.ones((10, 10))
            partner_main_dilate = cv2.dilate(partner_main, kernel, iterations=1)
            focus_main_dilate = cv2.dilate(focus_main, kernel, iterations=1)
            if np.sum(focus_main + partner_main_dilate == 2)>50:
                # Focus main 接壤 Partner main
                for  focus_sub in focus[1:]:
                    focus_sub_unique = focus_sub > partner_main
                    if np.sum(focus_sub_unique + partner_main_dilate==2)>50:
                        # Focus sub 接壤 Partner main, 保留
                        continue
                    elif np.sum(focus_sub_unique + focus_main_dilate==2)>50:
                        # Focus sub 接壤 Focus main, 保留
                        continue
                    else:
                        # Focus sub 不接壤 Partner sub, 删除
                        focus_sub[:] = 0
            else:
                # Focus main 不接壤 Partner main
                for  focus_sub in focus[1:]:
                    focus_sub_unique = focus_sub > partner_main
                    if np.sum(focus_sub_unique + focus_main_dilate==2)>50:
                        # Focus sub 接壤 Partner sub, 保留
                        continue
                    elif (np.sum(focus_sub_unique + partner_main_dilate==2)>50
                        and np.sum(focus_sub_unique + focus_main_dilate==2)==0):
                        # Focus sub 接壤 Partner sub, 移交给 Partner main
                        partner_main[:] = (partner_main + focus_sub >0).astype(partner_main.dtype)
                        focus_main_dilate = cv2.dilate(focus_main, kernel, iterations=1)
                        focus_sub[:] = 0
                    else:
                        # Focus sub 不接壤 Partner sub, 删除
                        focus_sub[:] = 0
    return masks0, masks1

def center_of_mass_cxf(input):
    a_x, a_y = np.sum(input, axis=0, keepdims=True), np.sum(input, axis=1, keepdims=True)
    a_all = np.sum(a_x)
    a_x, a_y = a_x/a_all, a_y/a_all
    grids = np.ogrid[[slice(0, i) for i in input.shape]]
    return np.sum(a_y*grids[0]), np.sum(a_x*grids[1])

def ims_to_com2ds(ims):
    coms_2d = []
    for im_mask in ims:
        assert im_mask.ndim == 2
        com_2d = center_of_mass_cxf(im_mask)[::-1] if np.max(im_mask) >= 1 else np.ones((2,))+np.nan
        coms_2d.append(com_2d)
    coms_2d = np.array(coms_2d)
    return coms_2d


def s1_filt_by_thr(result, thr=0.5):
    result_out = []
    for a_frame_result in result:
        bbox_results, mask_results = a_frame_result
        a_frame_out = [[],[]]
        for a_class_bbox, a_class_mask in zip(bbox_results, mask_results):
            p_vals = a_class_bbox[:,-1]
            valid  = p_vals > thr
            a_class_bbox = a_class_bbox[valid]
            a_class_mask = [mask for mask,v in zip(a_class_mask,valid) if v]
            a_frame_out[0].append(a_class_bbox)
            a_frame_out[1].append(a_class_mask)

        result_out.append(a_frame_out)
    return result_out

def s2_det2seg_part(resultf):
    nclass = len(resultf[0][0])
    masks0 = np.array(resultf[0][1][0], dtype='uint8')
    masks1 = np.array(resultf[0][1][1], dtype='uint8')
    masks0, masks1 = refine_mask(masks0, masks1)# nmasks, y, x
    pvals = []
    if len(masks0) and len(masks1):
        masks = np.concatenate([masks0, masks1], axis=0) > 0
    elif len(masks0) == 0:
        masks = masks1 > 0
    elif len(masks1) == 0:
        masks = masks0 > 0
    classids = []

    for iclass in range(nclass):
        for boxp, maskzip in zip(resultf[0][0][iclass], resultf[0][1][iclass]):
            pvals.append(boxp[-1])
            classids.append(iclass)

    canvas = np.zeros((800,1280))
    pvals = np.array(pvals)
    classids = np.array(classids)
    sort_inds = np.argsort(classids)[::-1]  # ascending order
    pvals = pvals[sort_inds]
    masks = masks[sort_inds]
    classids = classids[sort_inds]
    pvals_dict = dict(zip(classids, pvals))
    for mask_mat, iclass in zip(masks, classids):
        canvas[mask_mat] = iclass + 1

    # canvas = mask2resize(canvas, shape_base)
    for iclass in range(nclass):
        mask = canvas == (iclass + 1)
        if np.sum(mask) <= 4:
            # ignore this mask
            resultf[0][0][iclass] = []
        else:
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            pval = pvals_dict[iclass]
            resultf[0][0][iclass] = [np.array([x, y, x + w, y + h, pval])]

        resultf[0][1][iclass] = [np.array(mask[:,:,np.newaxis], order='F', dtype=np.uint8)]
    return resultf

kernel_size = (21, 21)
kernel_np = np.ones(kernel_size, np.uint8)

def s3_dilate_cv(result):

    nclass = len(result[0][0])
    for iclass in range(nclass):
        mask = result[0][1][iclass][0]
        out_array = cv2.dilate(mask, kernel_np, iterations=1)
        result[0][1][iclass] = [out_array]
    return result 

def s4_com2d(result):
    nclass = len(result[0][0])
    coms_real_2d = np.zeros((nclass, 2))
    for iclass in range(nclass):
        mask = result[0][1][iclass]
        coms_real_2d[iclass, :] = ims_to_com2ds(mask)
    return coms_real_2d


def model_output_detach(outputs):
    re_outputs = [garray.detach().cpu() for garray in outputs]
    return re_outputs  

def model_output_result(outputs, img_metas:dict, CLASSES):
    if outputs[0].get_device()>=0:
        outputs = model_output_detach(outputs)
    batch_num_dets, batch_boxes, batch_scores, batch_labels = outputs[:4]
    batch_dets = torch.cat(
        [batch_boxes, batch_scores.unsqueeze(-1)], dim=-1)
    batch_masks = None if len(outputs) < 5 else outputs[4]
    batch_size = 1
    results = []
    rescale = True
    i=0 #batch
    num_dets = batch_num_dets[i]
    dets, labels = batch_dets[i][:num_dets], batch_labels[i][:num_dets]
    old_dets = dets.clone()
    labels = labels.int()

    if rescale:
        scale_factor = img_metas['scale_factor']

        if isinstance(scale_factor, (list, tuple, np.ndarray)):
            assert len(scale_factor) == 4
            scale_factor = dets.new_tensor(scale_factor)[
                None, :]  # [1,4]
        dets[:, :4] /= scale_factor

    dets_results = bbox2result(dets, labels, len(CLASSES))
    assert batch_masks is not None
    masks = batch_masks[i][:num_dets].unsqueeze(1)
    num_classes = len(CLASSES)
    class_agnostic = True
    segms_results = [[] for _ in range(num_classes)]
    
    if num_dets>0:
        segms_results = FCNMaskHead.get_seg_masks(
            Addict(
                num_classes=num_classes,
                class_agnostic=class_agnostic),
            masks,
            old_dets,
            labels,
            rcnn_test_cfg=Addict(mask_thr_binary=0.5),
            ori_shape=img_metas['ori_shape'],
            scale_factor=scale_factor,
            rescale=rescale)
        
    results.append((dets_results, segms_results))
    

    return results


def create_segpkl(q:Queue, q2:Queue, img_metas:dict, CLASSES):

    result_all = []
    com2d_all = []
    iframes = []
    outdata = dict()
    while True:
        iframe, result = q.get()
        if iframe is None:
            break
        # continue
        result = s1_filt_by_thr(result)# video2detpkl part
        result = s2_det2seg_part(result)# detpkl2segpkl part
        result = s3_dilate_cv(result)  # segpkl dilate part
        coms_real_2d = s4_com2d(result)  # segpkl com2d part
    
        result_encode = [(bbox_results, encode_mask_results(mask_results))
        for bbox_results, mask_results in result]

        result_all.append(result_encode[0])
        com2d_all.append(coms_real_2d)
        iframes.append(iframe)

    outdata['segdata'] = result_all
    outdata['coms_2d'] = com2d_all
    outdata['iframes'] = iframes

    q2.put(outdata)



class MyWorker(mmap_cuda.Worker):
# class MyWorker():
    def __init__(self, config, checkpoint, maxlen):
        super().__init__()
        self.cuda = getattr(self, "cuda", 0)
        self.id = getattr(self, "id", 0)
        self.config = config
        self.checkpoint = checkpoint
        self.maxlen = maxlen
        print("well setup worker:", self.cuda)
        self.cworker_per_gpu = 4

    def compute(self, args):
        video, (iview, crop_xywh) = args
        out_pkl = osp.splitext(video)[0] + f"_{iview}.seg2pkl"
        if os.path.exists(out_pkl):
            print("Skipping:", osp.basename(out_pkl))
            outdata = pickle.load(open(out_pkl, 'rb'))

            return outdata
        
        with torch.cuda.device(self.cuda), torch.no_grad():
            model = create_wrap_detector(self.checkpoint, self.config, "cuda")
            # model = init_detector(self.config, self.checkpoint, 'cuda')
            img_metas = prefetch_img_metas(model.cfg, crop_xywh[2:])
            resize_wh = img_metas["pad_shape"][1::-1]

        vid = ffmpegcv.VideoCaptureNV(
            video,
            crop_xywh=crop_xywh,
            resize=resize_wh,
            gpu=int(self.cuda),
            pix_fmt="rgb24",
        )

        maxlen = min([len(vid), self.maxlen]) if self.maxlen else len(vid)
        context = mp.get_context()
        q = context.Queue(maxsize=100)
        q2 = context.Queue(maxsize=10)
        c_process = [context.Process(target=create_segpkl, args=(q,q2,img_metas,model.CLASSES)) for _ in range(self.cworker_per_gpu)]
        _ = [process.start() for process in c_process]

        with torch.cuda.device(self.cuda):
            # warm up
            frame = np.zeros((resize_wh[1],resize_wh[0],3),dtype=np.uint8)
            data = process_img(frame, img_metas)
            gpu_outputs = model.forward_test(data['img'])
            outputs = model_output_detach(gpu_outputs)
            result = model_output_result(outputs, img_metas, model.CLASSES) #
            result2 = model(return_loss=False, rescale=True, **data) # assert result2 == result

        with torch.cuda.device(self.cuda), torch.no_grad(), vid:
            for iframe in tqdm.trange(maxlen, position=self.id, desc=f"[{self.id}]"):
                ret, frame = vid.read() #0:n
                outputs = model_output_detach(gpu_outputs) #-1:n-1
                data = process_img(frame, img_metas) #0:n
                gpu_outputs= model.forward_test(data['img']) #0:n 
                result = model_output_result(outputs, img_metas, model.CLASSES) #-1:n-1
                if iframe>0:
                    q.put((iframe-1, result)) #-1:n-1

        outputs = model_output_detach(gpu_outputs) #n
        result = model_output_result(outputs, img_metas, model.CLASSES) #n
        q.put((iframe, result))
        vid.release()

        _ = [q.put((None, None)) for _ in range(self.cworker_per_gpu)]
        outdata_l = [q2.get() for _ in range(self.cworker_per_gpu)]
        outdata = {k:[] for k in outdata_l[0].keys()}
        for outdata_i in outdata_l:
            for k in outdata.keys():
                outdata[k].extend(outdata_i[k])

        _ = [process.join() for process in c_process]
        # 排序
        iframes = np.array(outdata['iframes'])
        idx = np.argsort(iframes)
        del outdata['iframes']
        for k in outdata.keys():
            outdata[k] = [outdata[k][i] for i in idx]
            
        # return 0
        # save  file
        pickle.dump(outdata, open(out_pkl, 'wb'))
        print('saved to', out_pkl)

        return out_pkl

def convert(vfile,nviews):

    pkl_files = glob.glob(osp.splitext(vfile)[0] + '_*.seg2pkl')
    p = re.compile('.*(\d+).seg2pkl$')
    views = [int(p.findall(pkl_file)[0]) for pkl_file in pkl_files]
    vinfo = get_info(vfile)
    assert len(views) == nviews
    assert len(views) == max(views)+1 and min(views) == 0 
    views_xywh = get_view_xywh_wrapper(len(views))

    segdata = [[] for _ in range(len(views))]
    com2d = [[] for _ in range(len(views))]

    for view, pkl_file in zip(views, pkl_files):
        data = pickle.load(open(pkl_file, 'rb'))
        segdata[view] = data['segdata']
        com2d[view] = data['coms_2d']

    outdata = { 'info': {
        'vfile': vfile, 
        'nview': len(views), 
        'fps': vinfo.fps,
        'vinfo': vinfo._asdict()},
    'views_xywh': views_xywh,
    'segdata': segdata,
    'coms_2d': com2d,
    'dilate_segdata': segdata,
    }
    # save  file
    outpkl  = osp.splitext(vfile)[0] + '.segpkl'
    pickle.dump(outdata, open(outpkl, 'wb'))
    print('saved to', outpkl)
    for iviews in range(nviews):
        os.remove(osp.splitext(vfile)[0]+f'_{iviews}.seg2pkl')
    return outpkl
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video_path", type=str, default=None, help="path to video or folder"
    )
    parser.add_argument('--pannels', type=int, default=9, help='crop views')
    parser.add_argument("--config", type=str, default=config)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--maxlen", type=int, default=None)

    args = parser.parse_args()

    views_xywh = get_view_xywh_wrapper(args.pannels)# 分割坐标 x轴 y轴 width height
    video_path = args.video_path
    assert osp.exists(video_path), "video_path not exists"
    if osp.isfile(video_path):
        videos_path = [video_path]
    elif osp.isdir(video_path):
        videos_path = [
            f
            for f in glob.glob(osp.join(video_path, "*.mp4"))
            if  'mask.mp4' not in f and 'smoothed' not in f
        ]
        assert len(videos_path) > 0, "no video found"
    else:
        raise ValueError("video_path is not a file or folder")
    if args.checkpoint is None:
        args.checkpoint = findcheckpoint_trt(args.config, "latest.trt")
        # args.checkpoint = findcheckpoint_trt(args.config, "latest.pth")


    print('total vfiles:', len(videos_path))
    num_gpus = min((torch.cuda.device_count(), len(video_path)*args.pannels))
    print("num_gpus:", num_gpus)
    # init the workers pool

    args_iterable = list(itertools.product(videos_path, enumerate(views_xywh)))
    
    mmap_cuda.workerpool_init(range(num_gpus), MyWorker, args.config, args.checkpoint, args.maxlen)
    detpkls = mmap_cuda.workerpool_compute_map(args_iterable)
    # worker = MyWorker(args.config, args.checkpoint, args.maxlen)
    # for args_ in args_iterable:
    #     outdata = worker.compute(args_)

    for vfile in videos_path:
        convert(vfile, args.pannels)
