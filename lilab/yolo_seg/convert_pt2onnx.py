import argparse
from io import BytesIO

import onnx
import torch
from ultralytics import YOLO

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import onnxsim
from torch import Tensor
from typing import Tuple

MODEL='/home/liying_lab/chenxinfeng/DATA/ultralytics/runs/segment/train7/weights/last.pt'


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


class PostSeg(nn.Module):
    export = True
    shape = None

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        proto = self.proto(x[0])  # mask protos
        bs = proto.shape[0]  # batch size
        mc = torch.cat(
            [self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)],
            2)  # mask coefficients
        mask_coeff = mc.transpose(1, 2)
        bboxes, scores = self.forward_det(x)
        return bboxes, scores, mask_coeff, proto

    def forward_det(self, x):
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
        boxes = boxes.softmax(-1) @ torch.arange(self.reg_max).to(boxes)
        boxes0, boxes1 = -boxes[:, :2, ...], boxes[:, 2:, ...]
        boxes = self.anchors.repeat(b, 2, 1) + torch.cat([boxes0, boxes1], 1)
        boxes = boxes * self.strides
        boxes = boxes.transpose(1, 2)
        scores = scores.transpose(1, 2)
        return boxes, scores


def optim(module: nn.Module):
    s = str(type(module))[6:-2].split('.')[-1]
    if s == 'Segment':
        setattr(module, '__class__', PostSeg)


class NormalizedModel(torch.nn.Module):
    def __init__(self, patchInnerHW, module):
        super().__init__()
        self.module = module
        self.patchH, self.patchW = patchInnerHW
        self.unfold = nn.Unfold(kernel_size=patchInnerHW, stride=patchInnerHW, padding=0)
        self.std_value = Parameter(torch.Tensor([255.0]), requires_grad=False)
        
    def forward(self, x): #x: NCHW rgb 0-255
        x = x/self.std_value
        patches = self.unfold(x[None,None])
        patches2 = patches.view(1, self.patchH, self.patchW, -1)
        patches3 = torch.permute(patches2, (3,0,1,2))

        if True:
            pad_value = 0.5
            input_HW = np.asarray(patches3.shape[-2:])
            innerHW_full = np.ceil(input_HW/32).astype(int)*32
            mask_HW = innerHW_full // 4
            pad_height, pad_width = (innerHW_full - input_HW) // 2
            mask_pad_width = pad_width // 4
            mask_pad_height = pad_height // 4
            slice_pad_height = slice(mask_pad_height, mask_HW[0]-mask_pad_height)
            slice_pad_width = slice(mask_pad_width, mask_HW[1]-mask_pad_width)
            patches3 = F.pad(patches3, (pad_width, pad_width, pad_height, pad_height), "constant", pad_value)
            xyxy_pad = torch.Tensor([pad_width, pad_height, pad_width, pad_height])

        imgNCHW = patches3.repeat(1, 3, 1, 1)
        outputs = self.module(imgNCHW)

        if True:
            outputs = list(outputs)
            outputs[0] = outputs[0] - xyxy_pad
            outputs[3] = outputs[3][:,:, slice_pad_height, slice_pad_width]
            outputs = tuple(outputs)

        return outputs
    

def main(args):
    YOLOv8 = YOLO(args.weights)
    device = 'cpu'
    model = YOLOv8.model.fuse().eval()
    for m in model.modules():
        optim(m)
        m.to(device)
    model.to(device)

    model = NormalizedModel((args.input_HW[0]//3, args.input_HW[1]//3), model)
    model.eval()

    fake_input = torch.randn(args.input_HW)
    output =  model(fake_input)
    save_path = args.weights.replace('.pt', '.full.onnx')
    with BytesIO() as f:
        torch.onnx.export(model,
                          fake_input,
                          f,
                          opset_version=11,
                          input_names=['images'],
                          output_names=['bboxes', 'scores',  
                                        'maskcoeff', 'proto'])
        f.seek(0)
        onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)

    if True:
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, 'assert check failed'
    onnx.save(onnx_model, save_path)
    print(f'ONNX export success, saved as {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        # required=True,
                        default=MODEL,
                        help='PyTorch yolov8 weights')
    parser.add_argument('--input-HW',
                        nargs='+',
                        type=int,
                        default=[400*3, 640*3],  #[640*3, 640*3]
                        help='Model input shape only for api builder')
    args = parser.parse_args()
    assert len(args.input_HW) == 2

    main(args)

