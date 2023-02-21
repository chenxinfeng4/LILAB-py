# model.pth -> model_full.onnx
python -m lilab.mmpose_dev.a2_convert_mmpose2onnx res50_coco_com2d_512x320_ZJF.py --full --dynamic

# model_full.onnx -> model_full.engine
trtexec --onnx=work_dirs/res50_coco_com2d_512x320_ZJF/latest.full.onnx \
--fp16 --saveEngine=work_dirs/res50_coco_com2d_512x320_ZJF/latest.full_fp16.engine \
--timingCacheFile=work_dirs/res50_coco_com2d_512x320_ZJF/.cache.txt \
--workspace=3072 --optShapes=input_1:4x3x320x512 \
--minShapes=input_1:1x3x320x512 --maxShapes=input_1:10x3x320x512


## Set variable 
# ball 文件
cd /home/liying_lab/chenxinfeng/DATA/mmpose
mfile='res50_coco_ball_512x320_cam9.py'
mfile_nake=${mfile%.*}

python -m lilab.mmpose_dev.a2_convert_mmpose2onnx $mfile --full --dynamic
trtexec --onnx=work_dirs/${mfile_nake}/latest.full.onnx \
--fp16 --saveEngine=work_dirs/${mfile_nake}/latest.full.engine \
--timingCacheFile=work_dirs/${mfile_nake}/.cache.txt \
--workspace=3072 --optShapes=input_1:4x3x320x512 \
--minShapes=input_1:1x3x320x512 --maxShapes=input_1:10x3x320x512
