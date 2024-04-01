#conda activate mmdet
vfile=/mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-3-27/2024-03-27_15-01-07
checkpoint=/home/liying_lab/chenxinfeng/DATA/ultralytics/work_dirs/yolov8_det_640x640_marmoset_0315/weights/last.singleton.engine

ball=/mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-1-31_sync/ballmove/ball_move_cam1.aligncalibpkl
#predict com2d
python -m lilab.yolo_det.s1_video2matpkl $vfile.mp4 --checkpoint $checkpoint --setupname frank
python -m lilab.yolo_det.iview6_nan $vfile.matpkl
#convert 2d -> 3d
python -m lilab.multiview_scripts_dev.s4_matpkl2matcalibpkl  $vfile.matpkl $ball  
python -m lilab.yolo_det.iview6_nan $vfile.matcalibpkl
#show
python -m lilab.multiview_scripts_dev.s5_show_calibpkl2video $vfile.matcalibpkl