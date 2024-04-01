# python -m lilab.yolo_det.convert_pt2onnx WEIGHT
#%%
from typing import Tuple
from io import BytesIO
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from torch.nn.parameter import Parameter
import argparse
import os.path as osp
import cv2
import itertools


IMG_PATH = '/home/liying_lab/chenxinfeng/DATA/ultralytics/data/ball_rg/ball_20230115/images/train/ball_move_cam1_003549.jpg'

INPUT_SHAPE = [1,3,480,640]


def singleton(outputs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input: boxes, scores, anchors, strides_vector. type=[torch.Tensor | np.ndarray]
    Output: boxes_, scores_
    ----
    boxes_ : shape = (nbatch,nclass,4), box = xyxy
    scores_ : shape = (nbatch,nclass)
    """
    if isinstance(outputs[0], np.ndarray):
        boxes, scores = outputs
    elif isinstance(outputs[0], torch.Tensor):
        boxes, scores = [o.cpu().numpy() for o in outputs]
    else:
        raise TypeError("Unsupported type for outputs: {}".format(type(outputs[0])))
    max_inds = np.argmax(scores, axis=1)
    scores_max = np.take_along_axis(scores, max_inds[...,None,:], axis=1).squeeze(1) #(nbatch,nclass)
    boxes0 = np.take_along_axis(boxes[...,None,:], max_inds[...,None,:,None], axis=1).squeeze(1) #(nbatch,nclass,4)
    return boxes0, scores_max


def nms(outputs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input: boxes, scores, anchors, strides_vector. type=[torch.Tensor | np.ndarray]
    Output: boxes_, scores_
    ----
    boxes_ : shape = (nbatch,nclass,4), box = xyxy
    scores_ : shape = (nbatch,nclass)
    """
    if isinstance(outputs[0], np.ndarray):
        boxes, scores = outputs
    elif isinstance(outputs[0], torch.Tensor):
        boxes, scores = [o.cpu().numpy() for o in outputs]
    else:
        raise TypeError("Unsupported type for outputs: {}".format(type(outputs[0])))
    
    #使用topk，预筛选更快
    N = 50
    indices = np.argpartition(scores, -N, axis=1)[:, -N:]
    scores_topk = np.take_along_axis(scores, indices, axis=1)
    boxes_topk = np.take_along_axis(boxes[...,None,:], indices[...,:,None], axis=1)
    nbatch, nclass = scores.shape[0], scores.shape[2]
    boxes0 = np.empty((nbatch, nclass), dtype=object)
    scores0 = np.empty((nbatch, nclass), dtype=object)
    num_ins = np.zeros((nbatch, nclass), dtype=int)
    score_threshold = 0.4
    threshold = 0.4

    for i,j in itertools.product(range(nbatch), range(nclass)):
        boxes_now = boxes_topk[i,:,j]
        confidence_scores = scores_topk[i,:,j]
        indices = cv2.dnn.NMSBoxes(bboxes=boxes_now, scores=confidence_scores, score_threshold=score_threshold, nms_threshold=threshold)
        num_ins[i,j] = len(indices)
        if num_ins[i,j] == 0:
            boxes0[i,j] = np.empty((0,4),dtype=np.float32)
            scores0[i,j] = np.empty((0,),dtype=np.float32)
        else:
            boxes0[i,j] = boxes_now[indices]
            scores0[i,j] = confidence_scores[indices]
    
    return boxes0, scores0, num_ins


def make_anchors(feats: Tensor,
                 strides: Tensor,
                 grid_cell_offset: float = 0.5) -> Tuple[Tensor, Tensor]:
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device,
                          dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device,
                          dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(
            torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def optim(module: nn.Module):
    s = str(type(module))[6:-2].split('.')[-1]
    if s == 'Detect':
        setattr(module, '__class__', PostDetect)


class NormalizedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model
        self.std_value = Parameter(torch.Tensor([255.0]), requires_grad=False)

    def forward(self, x):
        x = x / self.std_value
        return self._model(x)


class PostDetect(nn.Module):
    shape = None
    dynamic = False

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        shape = x[0].shape
        b, res, b_reg_num = shape[0], [], self.reg_max * 4
        for i in range(self.nl):
            res.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
        self.anchors, self.strides = (x.transpose(
            0, 1) for x in make_anchors(x, self.stride, 0.5))
        self.anchors = torch.from_numpy(self.anchors.detach().numpy())
        self.strides = torch.from_numpy(self.strides.detach().numpy())
        x = [i.view(b, self.no, -1) for i in res]
        y = torch.cat(x, 2)
        boxes, scores = y[:, :b_reg_num, ...], y[:, b_reg_num:, ...].sigmoid()
        boxes = boxes.view(b, 4, self.reg_max, -1).permute(0, 1, 3, 2)
        boxes = (boxes.softmax(-1) @ torch.arange(self.reg_max).to(boxes)[:,None])[...,0]
        boxes0, boxes1 = -boxes[:, :2, ...], boxes[:, 2:, ...]
        boxes = self.anchors.repeat(b, 2, 1) + torch.cat([boxes0, boxes1], 1)
        boxes = boxes * self.strides[:,None,:]

        return boxes.transpose(1, 2), scores.transpose(1, 2)

#%%
def image_demo(model, IMG_PATH, outimgfile='out2.jpg', normed=False, padding=True):
    import cv2
    COLORS = {0:(0, 255, 0), 1:(0, 0, 255)}

    if padding:
        img = np.zeros((640,640,3)).astype(np.uint8) + 122
        img_fg = cv2.imread(IMG_PATH)
        img[80:640-80,...] = img_fg
    else:
        img = cv2.imread(IMG_PATH)
    # img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
    img_rgb_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    M =  [np.float32([[1, 0, -100], [0, 1, -100]]),
          np.float32([[1, 0, 100], [0, 1, 100]]),
          np.float32([[0.83, 0.24, 0], [-0.24, 0.83, 0]]),
          np.float32([[0.83, -0.24, 0], [0.24, 0.83, 0]])]
    shifted_images = [cv2.warpAffine(img_rgb_.copy(), M_i, (img.shape[1], img.shape[0]))
                      for M_i in M]
    
    inputimgs = np.stack([img_rgb_]+shifted_images, axis=0)
    # 应用仿射变换
    img_rgb = torch.from_numpy(inputimgs).float().permute(0, 3, 1, 2)
    if not normed:
        img_rgb /= 255.0
    outputs = model(img_rgb)
    bboxes, scores = singleton(outputs)
    labels = [0,1]
    outimg = []
    bboxes = bboxes.astype(int)
    for img, bboxes_, scores_ in zip(inputimgs, bboxes, scores):
        img = img.copy()
        for i, (bbox, score, label) in enumerate(zip(bboxes_, scores_, labels)):
            (x, y, x2, y2) = bbox
            color = COLORS[label]
            cv2.rectangle(img, (x, y), (x2, y2), color, 2)
            text = '{:.2f}'.format(score)
            cv2.putText(img, text, (50, 50+int(i)*40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, color, 2)
        outimg.append(img)
    img_ = np.concatenate(outimg, axis=1)
    cv2.imwrite(outimgfile, img_)



def main():
    import onnx
    from ultralytics import YOLO
    import ffmpegcv
    weights = '/home/liying_lab/chenxinfeng/DATA/ultralytics/work_dirs/yolov8_det_640x640_marmoset/weights/last.pt'
    vid = ffmpegcv.VideoCapture('/mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-3-27/2024-03-27_15-01-07_cam1.mp4',
                                resize=(640, 480), resize_keepratio=False)
    ret, img = vid.read()
    IMG_PATH = 'a.jpg'
    cv2.imwrite(IMG_PATH, img)

    YOLOv8 = YOLO(weights)
    model = YOLOv8.model.fuse().eval()
    model.to('cpu')
    fake_input = torch.randn(INPUT_SHAPE)
    for m in model.modules(): optim(m)
    outputs = model(fake_input)
    model2 = NormalizedModel(model)
    outputs = model2(fake_input/255.0)
    image_demo(model2, IMG_PATH, normed=True, outimgfile='out_3.jpg', padding=False)


if __name__ == '__main__':
    main()
