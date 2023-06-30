# %% single file
vfile=/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/crop9/tmp

config=/home/liying_lab/chenxinfeng/DATA/CBNetV2/mask_rcnn_r101_fpn_2x_coco_bwrat_816x512_cam9_oldrat.py
python -m lilab.mmdet_dev.s1_mmdet_videos2pkl_trt $vfile.mp4 --pannels 9 --config $config

ls $vfile*.pkl | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s2_detpkl_to_segpkl
python -m lilab.mmdet_dev.s2_segpkl_merge $vfile.mp4
python -m lilab.mmdet_dev.s2_segpkl_dilate "$vfile.segpkl"
rm $vfile*.pkl
python -m lilab.mmdet_dev.s3_segpkl_dilate2videoseg_canvas_mask "$vfile.mp4" --maxlen 9000
python -m lilab.mmdet_dev.s3_segpkl_dilate2videoseg_canvas_mask "$vfile.mp4" --disable-dilate --maxlen 9000

ball=/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/crop9/2022-04-25ball_crop9.calibpkl
python -m lilab.mmdet_dev.s4_segpkl_put_com3d_pro "$vfile.segpkl" --calibpkl "$ball"

#process (optional)
python -m lilab.mmdet_dev.s4_segpkl_com3d_to_video "$vfile.mp4" --vox_size 170

# %% batch
vdir=/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75
config=/home/liying_lab/chenxinfeng/DATA/CBNetV2/mask_rcnn_r101_fpn_2x_coco_bwrat_816x512_cam9_oldrat.py
python -m lilab.mmdet_dev.s1_mmdet_videos2pkl_trt $vdir --pannels 9 --config $config # --maxlen 900
ls $vdir/*.pkl | xargs -n 1 -P 50 python -m lilab.mmdet_dev.s2_detpkl_to_segpkl
ls $vdir/*.mp4 | xargs -n 1 -P 10 python -m lilab.mmdet_dev.s2_segpkl_merge
rm $vdir/*.pkl
python -m lilab.mmdet_dev.s2_segpkl_dilate $vdir
python -m lilab.mmdet_dev.s3_segpkl_dilate2videoseg_canvas_mask $vdir --maxlen 9000

ball=`ls $vdir/ball/*.calibpkl`

ls $vdir/*.segpkl | xargs -n 1 -I {} -P 4 python -m lilab.mmdet_dev.s4_segpkl_put_com3d_pro {} --calibpkl "$ball"

#processor A (optional)
ls $vdir/*.segpkl | sed 's/.segpkl/.mp4/' | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s4_segpkl_com3d_to_video --vox_size 240
#processor B (optional)
ls $vdir/*.segpkl | sed 's/.segpkl/.mp4/' | grep 'F_' | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s4_segpkl_com3d_to_video --vox_size 150
ls $vdir/*.segpkl | sed 's/.segpkl/.mp4/' | grep 'M_' | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s4_segpkl_com3d_to_video --vox_size 170
ls $vdir/*.segpkl | sed 's/.segpkl/.mp4/' | grep 'M_' | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s4_segpkl_com3d_to_video --vox_size 190

#
# DAY75 F190
# DAY75 M220