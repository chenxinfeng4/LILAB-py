vfile=/mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-3-29/marmoset/twomarmoset
setupname=frank2
checkpoint=/home/liying_lab/chenxinfeng/DATA/ultralytics/work_dirs/yolov8_det_640x640_marmoset/weights/last.singleton.engine

python -m lilab.yolo_det.s1_video2matpkl_multianimal $vfile.mp4 --setupname $setupname --checkpoint $checkpoint

python -m lilab.multiview_scripts_dev.s5_show_calibpkl2video $vfile.matpkl