# 针对 sktdraw_smoothed_foot.mp4 生成 400p 视频
ls *sktdraw_smoothed_foot.mp4 | xargs -n 1 -P 4 -I {} ffmpeg -hwaccel cuda -vcodec h264_cuvid -resize 400x400 -i {} {}.400p.mp4 -y
ls 2023-05-28_14-05-40GBxCW*sktdraw_smoothed_foot.mp4  | xargs -n 1 -P 2 -I {} ffmpeg -hwaccel cuda -vcodec h264_cuvid -resize 400x400 -i {} {}.400p.mp4 -y
rename -f 's/_3_sktdraw_smoothed_foot.mp4./_1_/' *.400p.mp4
