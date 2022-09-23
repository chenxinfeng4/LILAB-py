# python -m lilab.mmdet.convert_mmdet2trt config.py
import os.path as osp
import subprocess
import mmdet2trt.mmdet2trt
from mmdet2trt.mmdet2trt import mmdet2trt
import torch

# config = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/mask_rcnn_r101_fpn_2x_coco_rat_oneclass.py'
config = '/home/liying_lab/chenxinfeng/DATA/mmdetection/mask_rcnn_r101_fpn_2x_coco_bwrat.py'
checkpoint = None

if checkpoint is None:
    config = osp.abspath(config)
    basedir = osp.dirname(config)
    basenakename = osp.splitext(osp.basename(config))[0]
    checkpoint = osp.join(basedir, 'work_dirs', basenakename, 'latest.pth')
    assert osp.isfile(checkpoint), 'checkpoint not found: {}'.format(checkpoint)

output = osp.splitext(checkpoint)[0] + '.trt'

subprocess.call(f'mmdet2trt "{config}" "{checkpoint}" "{output}" --fp16 True --enable-mask True', shell=True)
trt_model = mmdet2trt(
    config,
    checkpoint,
    fp16_mode=True,
    enable_mask=True)

print('Saving TRT model to: {}'.format(osp.basename(output)))
torch.save(trt_model.state_dict(), output)
