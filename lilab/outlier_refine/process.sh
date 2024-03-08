imgdir='/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/crop9/errorframe/w'
#imgdir='/mnt/liying.cibr.ac.cn_Data_Temp/LS_NAC_fiberphotometry/1stBatch_NAC/error_3d'
videodir='/mnt/liying.cibr.ac.cn_Data_Temp/LS_NAC_fiberphotometry/1stBatch_NAC/tmp1'
ratname='white'



# 0. rename
# 2022-10-14_13-59-22_1_sktdraw_smoothed_w16.mp4_20221025_155614.589.png  2022-10-14_13-59-22.mp4\M_bwtxwhet_apart
python -m lilab.outlier_refine.s0_errorframe_separate_by_keyboard $imgdir
rename 's/_1_sktdraw//' -d $imgdir/*.jpg $imgdir/*.png


## 1. Crop and then OCR the error frames
python -m lilab.outlier_refine.s0_pot_player_errorframe_crop $imgdir
ls -d $imgdir/*/out* | xargs -n 1 python -m lilab.outlier_refine.s1_pot_player_errorframe

## 如果挑选 segment 图片用于 voxel label
ls -d $imgdir/out*/out.json | xargs -P 2 -n 1 python -m lilab.cvutils_new.extract_frames_canvas_fromjson --dir_name $videodir --rat_name $ratname


## 如果挑选原始图片用于 segment label, 也可用于ball label
# %% segment error frames
ls -d $imgdir/*/out*/out.json | xargs -P 2 -n 1 python -m lilab.cvutils_new.extract_frames_fromjson --dir_name $videodir
