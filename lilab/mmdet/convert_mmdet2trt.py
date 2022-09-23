# python -m lilab.mmdet.convert_mmdet2trt #--config config.py
import os.path as osp
import subprocess
import mmdet2trt.mmdet2trt
from mmdet2trt.mmdet2trt import mmdet2trt
import torch
import argparse


# config = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/mask_rcnn_r101_fpn_2x_coco_rat_oneclass.py'
#config = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/mask_rcnn_r101_fpn_2x_coco_bwrat.py'
config = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/mask_rcnn_r101_fpn_2x_coco_bwrat_taoxm_20220705.py'

checkpoint = None

def find_checkpoint(config):
    basedir = osp.dirname(config)
    basenakename = osp.splitext(osp.basename(config))[0]
    checkpoint = osp.join(basedir, 'work_dirs', basenakename, 'latest.pth')
    assert osp.isfile(checkpoint), 'checkpoint not found: {}'.format(checkpoint)
    return checkpoint

def convert2trt(config, checkpoint):
    output = osp.splitext(checkpoint)[0] + '.trt'

    subprocess.call(f'mmdet2trt "{config}" "{checkpoint}" "{output}" --fp16 True --enable-mask True', shell=True)
    trt_model = mmdet2trt(
        config,
        checkpoint,
        fp16_mode=True,
        enable_mask=True)

    print('Saving TRT model to: {}'.format(osp.basename(output)))
    torch.save(trt_model.state_dict(), output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = find_checkpoint(args.config)
    convert2trt(args.config, args.checkpoint)
