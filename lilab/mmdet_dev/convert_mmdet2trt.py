# python -m lilab.mmdet_dev.convert_mmdet2trt config.py
import os.path as osp
import subprocess
import mmdet2trt.mmdet2trt
from mmdet2trt.mmdet2trt import mmdet2trt
import torch
import argparse
from lilab.mmpose_dev.a2_convert_mmpose2engine import findcheckpoint_pth

# config = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/mask_rcnn_r101_fpn_2x_coco_rat_oneclass.py'
# config = '/home/liying_lab/chenxinfeng/DATA/mmdetection/mask_rcnn_r101_fpn_2x_coco_bwrat.py'


def convert(config, checkpoint):
    output = osp.splitext(checkpoint)[0] + '.trt'
    # subprocess.call(f'mmdet2trt "{config}" "{checkpoint}" "{output}" --fp16 True --enable-mask True', shell=True)
    trt_model = mmdet2trt(
        config,
        checkpoint,
        fp16_mode=True,
        enable_mask=True)
    print('Saving TRT model to: {}'.format(osp.basename(output)))
    torch.save(trt_model.state_dict(), output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert mmdet model to TensorRT model')
    parser.add_argument('config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file', default=None)
    args = parser.parse_args()
    config = args.config
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = findcheckpoint_pth(config)
    convert(config, checkpoint)
