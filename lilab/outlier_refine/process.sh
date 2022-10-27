imgdir='/mnt/ftp.rat/multiview_large/lhj_20221014/errorframe/white_groom'
videodir='/mnt/ftp.rat/multiview_large/lhj_20221014'
ratname='white'
outframes_dir=$videodir/outframes
outframes_ratdir=${outframes_dir}_${ratname}


# 0. rename
rename 's/_1_sktdraw_smoothed_w16//' -d $imgdir/*.jpg $imgdir/*.png

## 1. Crop and then OCR the error frames
python -m lilab.outlier_refine.s0_pot_player_errorframe_crop $imgdir
ls -d $imgdir/out* | xargs -n 1 python -m lilab.outlier_refine.s1_pot_player_errorframe
ls -d $imgdir/out*/out.json | xargs -P 2 -n 1 python -m lilab.cvutils_new.extract_frames_canvas_fromjson --dir_name $videodir --rat_name $ratname

