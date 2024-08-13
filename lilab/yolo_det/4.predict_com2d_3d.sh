#conda activate mmdet
vfile=/mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-4-13-singlemarmoset/output_test_dannce/output_640x480
checkpoint=/home/liying_lab/chenxinfeng/DATA/ultralytics/work_dirs/yolov8_det_640x640_marmoset/weights/last.singleton.engine
ball='/mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-4-14_balltrack/ballmove_convert_mm/2024-04-14_21-28-40.calibpkl'
#ball='/mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-4-14_balltrack/ballmove_convert_mm/2024-04-14_21-28-40.mkv.calibpkl'
#predict com2d
python -m lilab.yolo_det.s1_video2matpkl $vfile.mp4 --checkpoint $checkpoint --setupname frank
python -m lilab.yolo_det.iview6_nan $vfile.matpkl
python -m lilab.multiview_scripts_dev.s5_show_calibpkl2video $vfile.matpkl

#convert 2d -> 3d
#
python -m lilab.multiview_scripts_dev.s4_matpkl2matcalibpkl $vfile.matpkl $ball  
python -m lilab.yolo_det.iview6_nan $vfile.matcalibpkl
#show
python -m lilab.multiview_scripts_dev.s5_show_calibpkl2video $vfile.matcalibpkl
#vfile=/mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-3-6/2024-03-06_15-22-55.matcalibpkl
#marmoset_camera3_cxf/2024-4-13-singlemarmoset/output
#压缩文件：ffmpeg -i /mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-4-13-singlemarmoset/output.mp4 -vf "scale=1920:960" /mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-4-13-singlemarmoset/output_3.mp4
#delete iview 5 and head data in matcalibpkl and produce segpkl
python -m lilab.dannce_single.s1_matcalibpkl_com3d_to_segpkl $vfile.matcalibpkl


# \\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\WSY\Pair_bonding\Before cohousing\20240402_Dora_Kelvin
# ffmpeg -i "/mnt/liying.cibr.ac.cn_Data_Temp/WSY/Pair_bonding/Before_cohousing/20240402_Dora_Kelvin/2024-04-02_15-01-31_cam5.mkv" -c:v copy -c:a copy "/mnt/liying.cibr.ac.cn_Data_Temp/Chenxinfeng/WSY/Pair_bonding/Before_cohousing/20240402_Dora_Kelvin/2024-04-02_15-01-31_cam5B.mp4"