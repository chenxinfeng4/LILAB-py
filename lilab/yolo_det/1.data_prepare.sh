#source activate mmdet
labelme_dir=`w2l '\\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\marmoset_camera3_cxf\2024-4-13-singlemarmoset\outframes'`
labelme_dir=`w2l '\\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\marmoset_camera3_cxf\2024-3-26\marmoset\labeled'`
target_dir='/home/liying_lab/chenxinfeng/DATA/ultralytics/data/marmoset_bodyhead/marmoset_20240617'

rm `grep -L 'points' $labelme_dir/*.json`
rm `comm -3 <(ls -1 $labelme_dir/*.jpg | sort) <(ls -1 $labelme_dir/*.json | sed s/json/jpg/ | sort)`

python -m lilab.yolo_det.labelme_to_yolo_det --json_dir $labelme_dir --val_size 0.2

mkdir -p $target_dir
cp -r $labelme_dir/YOLODataset/* $target_dir

header="train: $target_dir/images/train/
val: $target_dir/images/val/"

cat <(echo "$header") <(sed '1,3d' $target_dir/dataset.yaml) > $target_dir/.dataset.yaml
mv $target_dir/.dataset.yaml $target_dir/dataset.yaml
cat $target_dir/dataset.yaml
