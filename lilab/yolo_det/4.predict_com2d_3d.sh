#conda activate mmdet
vfile=/mnt/liying.cibr.ac.cn_usb3/wsy/view15/20241125_15view_6id/Iris/dehead/2024-11-25_18-42-30_dehead
checkpoint=/home/liying_lab/chenxinfeng/DATA/ultralytics/work_dirs/yolov8n_det_640_marmoset/weights/last.singleton.engine

# ball=/mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-1-31_sync/ballmove/ball_move_cam1.aligncalibpkl
# ball=/home/liying_lab/chenxf/ml-project/syncdata/ball_move.aligncalibpkl
ball=/home/liying_lab/chenxf/ml-project/论文图表/狨猴cam_calib/example.calibpkl
#predict com2d
python -m lilab.yolo_det.s1_video2matpkl $vfile.mp4 --checkpoint $checkpoint --setupname gavin --headerviews 5
# python -m lilab.yolo_det.iview6_nan $vfile.matpkl
#convert 2d -> 3d
python -m lilab.multiview_scripts_dev.s4_matpkl2matcalibpkl  $vfile.matpkl $ball --fix-camnum
# python -m lilab.yolo_det.iview6_nan $vfile.matcalibpkl
#show
python -m lilab.yolo_det.s5_show_calibpkl2video $vfile.matcalibpkl
# python -m lilab.multiview_scripts_dev.s5_show_calibpkl2video $vfile.matcalibpkl

#python -m lilab.multiview_scripts_dev.s4_matpkl2matcalibpkl_image  $vfile.matpkl $ball  