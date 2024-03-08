#!/bin/bash
#create_400p.sh /A/B/C/
# 针对 sktdraw_smoothed_foot.mp4 生成 400p 视频
cd $1  #切换到项目目录

# 单 GPU 版本
# ls *sktdraw_smoothed_foot..mp4 | xargs -n 1 -P 4 -I {} ffmpeg -hwaccel cuda -vcodec hevc_cuvid -resize 1280x800 -i {} {}.400p.mp4 -y
# rename -f 's/_3_sktdraw_smoothed_foot.mp4./_1_/' *.400p.mp4
#/mnt/liying.cibr.ac.cn_Data_Temp/ZYQ_data_temp/CPP/Behavior

# 多 GPU 版本
ls *sktdraw_smoothed_foot.mp4 > .tmp
total_rows=$(wc -l < .tmp)
rows_per_variable=$(echo "scale=0; ($total_rows + 3) / 4" | bc)
split -l "$rows_per_variable" .tmp files_

cat files_ad | xargs -n 1 -P 2 -I {} ffmpeg -hwaccel_device 0 -hwaccel cuda -vcodec h264_cuvid -resize 400x400 -i {} {}.400p.mp4 -y &
cat files_ab | xargs -n 1 -P 2 -I {} ffmpeg -hwaccel_device 1 -hwaccel cuda -vcodec h264_cuvid -resize 400x400 -i {} {}.400p.mp4 -y &
cat files_ac | xargs -n 1 -P 2 -I {} ffmpeg -hwaccel_device 2 -hwaccel cuda -vcodec h264_cuvid -resize 400x400 -i {} {}.400p.mp4 -y &
cat files_aa | xargs -n 1 -P 2 -I {} ffmpeg -hwaccel_device 3 -hwaccel cuda -vcodec h264_cuvid -resize 400x400 -i {} {}.400p.mp4 -y &

wait
rename -f -E 's/(_[0-9])?_sktdraw_smoothed_foot.mp4./_1_/' *.400p.mp4
rm .tmp files_a*
