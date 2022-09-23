imgdir='/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/TPH2KOxWT/errorframes/black'
videodir='/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/TPH2KOxWT/tmp_video'
ratname='black'
outframes_dir=$videodir/outframes
outframes_ratdir=${outframes_dir}_${ratname}

## 1. Crop and then OCR the error frames
python -m lilab.outlier_refine.s0_pot_player_errorframe_crop $imgdir
ls -d $imgdir/out* | xargs -n 1 python -m lilab.outlier_refine.s1_pot_player_errorframe
ls -d $imgdir/out*/out.json | xargs -n 1 sed 's/_1_sktdraw//g' -i
ls -d $imgdir/out*/out.json | xargs -P 0 -n 1 python -m lilab.cvutils_new.extract_frames_canvas_fromjson --dir_name $videodir

## 3. Move to the correct folder
mv $outframes_dir/* $outframes_ratdir

## 4. Remove the other rat
ls -d $outframes_ratdir/*.jpg | grep -v _rat$ratname | xargs rm
