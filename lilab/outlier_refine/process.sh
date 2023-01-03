imgdir='/home/liying_lab/chenxinfeng/ftp.image_ZYQ/multiview_large/errorframe/errorframe20230101/b'
videodir='/home/liying_lab/chenxinfeng/ftp.image_ZYQ/multiview_large/animal_only/20221124'
ratname='black'


# 0. rename
# 2022-10-14_13-59-22_1_sktdraw_smoothed_w16.mp4_20221025_155614.589.png  2022-10-14_13-59-22.mp4\M_bwtxwhet_apart
rename 's/_1_sktdraw//' -d $imgdir/*.jpg $imgdir/*.png

## 1. Crop and then OCR the error frames
python -m lilab.outlier_refine.s0_pot_player_errorframe_crop $imgdir
ls -d $imgdir/out* | xargs -n 1 python -m lilab.outlier_refine.s1_pot_player_errorframe

## 如果挑选 segment 图片用于 voxel label
ls -d $imgdir/out*/out.json | xargs -P 2 -n 1 python -m lilab.cvutils_new.extract_frames_canvas_fromjson --dir_name $videodir --rat_name $ratname


## 如果挑选原始图片用于 segment label
# %% segmenta error frames
ls -d $imgdir/out*/out.json | xargs -P 2 -n 1 python -m lilab.cvutils_new.extract_frames_fromjson --dir_name $videodir
