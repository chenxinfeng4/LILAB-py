# model.pth -> model_full.onnx
# ball 文件 或者com2d文件
mfile='res50_coco_ball_512x512_ZYY.py'

cd /home/liying_lab/chenxinfeng/DATA/mmpose
mfile_nake=${mfile%.*}

python -m lilab.mmpose_dev.a2_convert_mmpose2onnx $mfile --full --dynamic
trtexec --onnx=work_dirs/${mfile_nake}/latest.full.onnx \
--fp16 --saveEngine=work_dirs/${mfile_nake}/latest.full.engine \
--timingCacheFile=work_dirs/${mfile_nake}/.cache.txt \
--workspace=3072 --optShapes=input_1:4x3x320x512 \
--minShapes=input_1:1x3x320x512 --maxShapes=input_1:10x3x320x512



mfile='res50_coco_com2d_352x256_ZZC_lbs.py'

cd /home/liying_lab/chenxinfeng/DATA/mmpose
mfile_nake=${mfile%.*}

python -m lilab.mmpose_dev.a2_convert_mmpose2onnx $mfile --full --dynamic
trtexec --onnx=work_dirs/${mfile_nake}/latest.full.onnx \
--fp16 --saveEngine=work_dirs/${mfile_nake}/latest.full.engine \
--timingCacheFile=work_dirs/${mfile_nake}/.cache.txt \
--workspace=3072 --optShapes=input_1:4x3x256x352 \
--minShapes=input_1:1x3x256x352 --maxShapes=input_1:10x3x256x352

python -m lilab.mmpose_dev.a2_convert_mmpose2onnx $mfile --full --dynamic --monocolor
trtexec --onnx=work_dirs/${mfile_nake}/latest.full.onnx \
--fp16 --saveEngine=work_dirs/${mfile_nake}/latest.full_mono.engine \
--timingCacheFile=work_dirs/${mfile_nake}/.mono.cache.txt \
--workspace=3072 --optShapes=input_1:4x1x256x352 \
--minShapes=input_1:1x1x256x352 --maxShapes=input_1:10x1x256x352


# ZYY project
mfile='res50_coco_ball_512x512_ZYY.py'

cd /home/liying_lab/chenxinfeng/DATA/mmpose
mfile_nake=${mfile%.*}

python -m lilab.mmpose_dev.a2_convert_mmpose2onnx $mfile --full --dynamic
trtexec --onnx=work_dirs/${mfile_nake}/latest.full.onnx \
    --fp16 --saveEngine=work_dirs/${mfile_nake}/latest.full.engine \
    --timingCacheFile=work_dirs/${mfile_nake}/.cache.txt \
    --workspace=3072 --optShapes=input_1:4x3x512x512 \
    --minShapes=input_1:1x3x512x512 --maxShapes=input_1:10x3x512x512