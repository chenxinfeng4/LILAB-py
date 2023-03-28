# 环境变量
conda activate mmpose
cd /home/liying_lab/chenxinfeng/DATA/mmpose/

config=res50_coco_ball_512x320_cam9.py
config_nake=`echo $config | sed 's/.py//'`
balllabeldir=/home/liying_lab/chenxinfeng/DATA/mmpose/data/ball/ball_1280x800x9_20230201

# 转换数据到 mmpose
python -m lilab.cvutils.labelme_to_cocokeypoints_ball $balllabeldir

# 修改模型
vim $config

# 开始训练，转换模型
tools/dist_train.sh $config 4

python -m lilab.mmpose_dev.a2_convert_mmpose2onnx $config --full --dynamic

trtexec --onnx=work_dirs/$config_nake/latest.full.onnx \
    --fp16 --saveEngine=work_dirs/$config_nake/latest.full.engine \
    --timingCacheFile=work_dirs/$config_nake/.cache.txt \
    --workspace=3072 --optShapes=input_1:9x3x320x512 \
    --minShapes=input_1:1x3x320x512 --maxShapes=input_1:10x3x320x512
