vfile=/home/liying_lab/chenxinfeng/DATA/rat_shank3/2022-10-10_14-15-16FbHETxwwt
python -m lilab.mmdet_dev.s1_mmdet_videos2pkl_trt $vfile.mp4 --pannels 9
ls $vfile*.pkl | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s2_detpkl_to_segpkl
python -m lilab.mmdet_dev.s2_segpkl_merge $vfile.mp4
python -m lilab.mmdet_dev.s2_segpkl_dilate "$vfile.segpkl"
python -m lilab.mmdet_dev.s3_segpkl_dilate2videoseg_canvas_mask "$vfile.mp4"
python -m lilab.mmdet_dev.s3_segpkl_dilate2videoseg_canvas_mask "$vfile.mp4" --disable-dilate


vfile=tmp_video/2022-06-16_16-56-23bkoxwko
ball=ball/2022-06-23ball.calibpkl
python -m lilab.mmdet_dev.s4_segpkl_put_com3d_pro "$vfiles.segpkl" --calibpkl "$ball"
python -m lilab.mmdet_dev.s4_segpkl_com3d_to_video "$vfile.mp4"


# %% batch
vdir=.
python -m lilab.mmdet_dev.s1_mmdet_videos2pkl_trt $vdir --pannels 9
ls $vdir/*.pkl | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s2_detpkl_to_segpkl
ls $vdir/*.mp4 | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s2_segpkl_merge
rm $vdir/*.pkl
python -m lilab.mmdet_dev.s2_segpkl_dilate $vdir
python -m lilab.mmdet_dev.s3_segpkl_dilate2videoseg_canvas_mask $vdir

ball=$vdir/2022-10-11_17-19-12ball.calibpkl
ls $vdir/*.segpkl | xargs -n 1 -I {} -P 0 python -m lilab.mmdet_dev.s4_segpkl_put_com3d_pro {} --calibpkl "$ball"
ls $vdir/*.mp4 | grep -v _mask | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s4_segpkl_com3d_to_video 
