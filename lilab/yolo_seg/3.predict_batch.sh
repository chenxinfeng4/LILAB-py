
# %% batch
# step0
# conda activate mmdet
source activate mmdet
vdir=`w2l '\\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\multiview_9\chenxf\test\2022-10-13_15-08-49AWxCB.mp4'`
ball=`w2l "\\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\multiview_9\chenxf\carl\2023-10-14-\ball_2023-10-23_13-18-10.calibpkl"`
checkpoint=/home/liying_lab/chenxinfeng/DATA/ultralytics/work_dirs/yolov8n_seg_832_ratbw/weights/last.full.engine
python -m lilab.yolo_seg.s1_mmdet_videos2segpkl_dilate $vdir --checkpoint $checkpoint #--maxlen 9000
python -m lilab.mmdet_dev.s3_segpkl_dilate2videoseg_canvas_mask $vdir  # check video


# step2 com3d and create voxbox
ls $vdir/*.segpkl | xargs -n 1 -I {} -P 4 python -m lilab.mmdet_dev.s4_segpkl_put_com3d_pro {} --calibpkl "$ball"
ls $vdir/*.segpkl | sed 's/.segpkl/.mp4/' | xargs -n 1 -P 8 -I {} python -m lilab.mmdet_dev.s4_segpkl_com3d_to_video {} --vox_size 230  # check video



# vfile=/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/LTZxWT_230505/multi/group2/2023-05-25_14-04-42EWxGB.mp4

# python -m lilab.mmdet_dev_multi.s1_mmdet_videos2segpkl_dilate $vfile --pannels 9 --config $config 

# python -m lilab.mmdet_dev.s4_segpkl_put_com3d_pro $vfile --calibpkl "$ball"
