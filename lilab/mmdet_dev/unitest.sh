#!/bin/bash
conda activate mmdet
choosecuda 1

config=/home/liying_lab/chenxinfeng/DATA/mmdetection/mask_rcnn_r101_fpn_2x_coco_bwrat.py
checkpoint=/home/liying_lab/chenxinfeng/DATA/CBNetV2/work_dirs/mask_rcnn_r101_fpn_2x_coco_bwrat/latest.pth
output=/home/liying_lab/chenxinfeng/DATA/CBNetV2/work_dirs/mask_rcnn_r101_fpn_2x_coco_bwrat/latest.trt


vdir=/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/TPH2KOxWT
vfile=2022-06-16_16-05-29.mp4

# 1. mmdet2trt
mmdet2trt $config $checkpoint $output --enable-mask True --max-workspace-gb 8 --fp16 True

# 2. mmdet trt video inference
python -m lilab.mmdet_dev.s1_mmdet_videos2pkl_trt_1280x800x10 $pdir/$vid --config $config --checkpoint $output

# 3. mmdet pkl to segpkl
ls $pdir/*.pkl | xargs -n 1 -P 10 python -m lilab.mmdet_dev.s2_detpkl_to_segpkl
python -m lilab.mmdet_dev.s2_segpkl_merge $pdir/$vid
ls $pdir/*segpkl | xargs -n 1 python -m lilab.mmdet_dev.s2_segpkl_dilate
python -m lilab.mmdet_dev.s3_segpkl2video_1280x800x10 $pdir/$vid

python -m lilab.mmdet_dev.s3_segpkl_dilate2videoseg_canvas $pdir/$vid
