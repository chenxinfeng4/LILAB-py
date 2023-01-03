# vfile=/mnt/ftp.rat/multiview_9/SHANK3HETxWT/2022-10-10/2022-10-10_14-15-16FbHETxwwt
vfile=/home/liying_lab/chenxinfeng/ftp.image_ZYQ/multiview_large/animal_only/20221125/2022-11-25_19-55-41_female_est_rat1

python -m lilab.mmdet_dev.s1_mmdet_videos2pkl_trt $vfile.mp4 --pannels 9
ls $vfile*.pkl | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s2_detpkl_to_segpkl
python -m lilab.mmdet_dev.s2_segpkl_merge $vfile.mp4
python -m lilab.mmdet_dev.s2_segpkl_dilate "$vfile.segpkl"
python -m lilab.mmdet_dev.s3_segpkl_dilate2videoseg_canvas_mask "$vfile.mp4"
python -m lilab.mmdet_dev.s3_segpkl_dilate2videoseg_canvas_mask "$vfile.mp4" --disable-dilate


vfile=tmp_video/2022-06-16_16-56-23bkoxwko
ball=ball/2022-06-23ball.calibpkl
ball=/home/liying_lab/chenxinfeng/ftp.image_ZYQ/multiview_large/animal_only/20221124/2022-11-24_16-36-57ball.calibpkl
python -m lilab.mmdet_dev.s4_segpkl_put_com3d_pro "$vfiles.segpkl" --calibpkl "$ball"
python -m lilab.mmdet_dev.s4_segpkl_com3d_to_video "$vfile.mp4"  --vox_size 190


# %% batch
vdir=/home/liying_lab/chenxinfeng/ftp.image_ZYQ/multiview_large/animal_only/20221126
python -m lilab.mmdet_dev.s1_mmdet_videos2pkl_trt $vdir --pannels 9
ls $vdir/*.pkl | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s2_detpkl_to_segpkl
ls $vdir/*.mp4 | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s2_segpkl_merge
rm $vdir/*.pkl
python -m lilab.mmdet_dev.s2_segpkl_dilate $vdir
python -m lilab.mmdet_dev.s3_segpkl_dilate2videoseg_canvas_mask $vdir

vdir=/home/liying_lab/chenxinfeng/ftp.image_ZYQ/multiview_large/animal_only/20221126
ball=$vdir/2022-11-24_19-11-47ball.calibpkl
ls $vdir/*.segpkl | xargs -n 1 -I {} -P 0 python -m lilab.mmdet_dev.s4_segpkl_put_com3d_pro {} --calibpkl "$ball"

#processor A
ls $vdir/*.segpkl | sed 's/.segpkl/.mp4/' | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s4_segpkl_com3d_to_video --vox_size 190
#processor B
ls $vdir/*.segpkl | sed 's/.segpkl/.mp4/' | grep 'F_' | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s4_segpkl_com3d_to_video --vox_size 150
ls $vdir/*.segpkl | sed 's/.segpkl/.mp4/' | grep 'M_' | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s4_segpkl_com3d_to_video --vox_size 170
ls $vdir/*.segpkl | sed 's/.segpkl/.mp4/' | grep 'M_' | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s4_segpkl_com3d_to_video --vox_size 190
