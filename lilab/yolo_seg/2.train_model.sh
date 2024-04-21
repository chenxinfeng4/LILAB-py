source activate open-mmlab


# 修改里面的 data 字段
cd /home/liying_lab/chenxinfeng/DATA/ultralytics
config_py=/home/liying_lab/chenxinfeng/DATA/ultralytics/yolov8n_seg_832_ratbw.py
echo $config_py
# vim $config_py


# 运行
python $config_py
