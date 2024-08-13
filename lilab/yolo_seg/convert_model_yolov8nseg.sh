#!/bin/bash
project_file=/home/liying_lab/chenxinfeng/DATA/ultralytics/work_dirs/yolov8n_seg_832_ratbw/weights/last

# 1 to ONNX
source activate open-mmlab
python /home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/yolo_seg/convert_pt2onnx.py --weights ${project_file}.pt --input-HW 1200 1920 #1632 2496

siinn inspect ${project_file}.full.onnx

# 2A
source activate mmdet
trtexec --onnx=${project_file}.full.onnx --fp16 --saveEngine=${project_file}.full.engine --timingCacheFile=${project_file}.full.cache
