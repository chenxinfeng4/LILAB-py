#!/bin/bash
project_file=/home/liying_lab/chenxinfeng/DATA/ultralytics/work_dirs/yolov8n_seg_640_ratbw/weights/last

# 1 to ONNX
source activate open-mmlab
python /home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/yolo_seg/convert_pt2onnx.py --weights ${project_file}.pt --input-HW 1200 1920

siinn inspect ${project_file}.full.onnx

# 2A
source activate mmdet
trtexec --onnx=${project_file}.full.onnx --fp16 --saveEngine=${project_file}.full.engine --timingCacheFile=${project_file}.full.cache
#trtexec --onnx=${project_file}.full.onnx --saveEngine=${project_file}.full.fp32.engine --timingCacheFile=${project_file}.fp32.full.cache

# 2B RKNN
source activate rknn
# python /home/liying_lab/chenxinfeng/ml-project/rknn-toolkit2/yolov8_det/convert_onnx2rknn.py \
#     --onnx ${project_file}.onnx --saveEngine $RKNN_MODEL
python /home/liying_lab/chenxinfeng/ml-project/rknn-toolkit2/yolov8_det/convert_onnx2rknn_simpley.py \
    --onnx ${project_file}.onnx --saveEngine $RKNN_MODEL
cp ${project_file}.anchorpkl $RKNN_MODEL.anchorpkl

# 香橙派
sudo su
source /home/chenxinfeng/.bashrc
source activate rknn
cd /home/chenxinfeng/ml-project/rknn_yolo


cp /mnt/ssh3090_mmpose/yolov8_det_640x480.rknn . ; python test.py 1> /dev/null
