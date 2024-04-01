source activate open-mmlab

# checkpt=/home/liying_lab/chenxinfeng/DATA/ultralytics/work_dirs/yolov8n_det_640_ballrg/weights/last.pt
checkpt=/home/liying_lab/chenxinfeng/DATA/ultralytics/work_dirs/yolov8_det_640x640_marmoset/weights/last.pt
# checkpt=/home/liying_lab/chenxinfeng/DATA/ultralytics/work_dirs/yolov8l_det_640_ballrg/weights/last.pt
checknake=${checkpt%.pt}

python -m lilab.yolo_det.convert_pt2onnx $checkpt --dynamic

source activate mmdet

trtexec --onnx=${checknake}.singleton.onnx \
    --fp16 --workspace=3072 --saveEngine=${checknake}.singleton.engine \
    --timingCacheFile=`dirname ${checknake}`/.cache.txt \
    --optShapes=input_1:5x3x480x640 \
    --minShapes=input_1:1x3x480x640 \
    --maxShapes=input_1:9x3x480x640


# trtexec --onnx=${checknake}.singleton.onnx \
#     --fp16 --workspace=3072 --saveEngine=${checknake}.singleton.engine \
#     --timingCacheFile=`dirname ${checknake}`/.cache.txt
