target_dir='/home/liying_lab/chenxinfeng/DATA/ultralytics/data/rats_metric/'
mkdir -p $target_dir

json_dir=/DATA/chenxinfeng/CBNetV2/data/rats_metric/tr1/tr1_all
python -m lilab.yolo_seg.labelme_to_yolo --json_dir $json_dir --val_size 0.0 --seg
cp -rf $json_dir/YOLODataset_seg/* $target_dir/

json_dir=/DATA/chenxinfeng/CBNetV2/data/rats_metric/te2/intense_pannel
python -m lilab.yolo_seg.labelme_to_yolo --json_dir $json_dir --val_size 1.0 --seg
cp -rf $json_dir/YOLODataset_seg/* $target_dir/



header="train: $target_dir/images/train/
val: $target_dir/images/val/"

cat <(echo "$header") <(sed '1,3d' $target_dir/dataset.yaml) > $target_dir/.dataset.yaml
mv $target_dir/.dataset.yaml $target_dir/dataset.yaml
cat $target_dir/dataset.yaml

ls $target_dir/dataset.yaml