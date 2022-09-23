vfile=/DATA/chenxinfeng/tao_rat/data/20220613-side6-addition/TPH2-KO-multiview-202201/male/cxf_batch/bwt-wwt-01-17_11-04-25

python -m lilab.mmdet_dev.s1_mmdet_videos2pkl_trt $vfile.mp4
ls $pdir/$vfile*pkl | xargs -n 1 -P 6 python -m lilab.mmdet_dev.s2_detpkl_to_segpkl
python -m lilab.mmdet_dev.s2_segpkl_merge $vfile.mp4
python -m lilab.mmdet_dev.s2_segpkl_dilate "$vfile.segpkl"
python -m lilab.mmdet_dev.s3_segpkl_dilate2videoseg_canvas_mask "$vfile.mp4"
python -m lilab.mmdet_dev.s3_segpkl_dilate2videoseg_canvas_mask "$vfile.mp4" --disable-dilate


vfile=tmp_video/2022-06-16_16-56-23bkoxwko
ball=ball/2022-06-23ball.calibpkl
python -m lilab.mmdet_dev.s4_segpkl_put_com3d_pro "$vfile.segpkl" --calibpkl "$ball"
python -m lilab.mmdet_dev.s4_segpkl_com3d_to_video "$vfile.mp4"
