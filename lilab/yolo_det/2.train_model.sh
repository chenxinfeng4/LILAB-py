source activate open-mmlab


# 修改里面的 data 字段
cd /home/liying_lab/chenxinfeng/DATA/ultralytics
# config_py=/home/liying_lab/chenxinfeng/DATA/ultralytics/yolov8_det_640x640_marmoset.py
config_py=/home/liying_lab/chenxinfeng/DATA/ultralytics/yolov8n_det_640_marmoset.py
#如果你要在此基础加训，直接进入原来的config_py改data，data = "/home/liying_lab/chenxinfeng/DATA/ultralytics/marmoset_dataset.yaml"里
echo $config_py
# vim $config_py


# 运行
python $config_py
