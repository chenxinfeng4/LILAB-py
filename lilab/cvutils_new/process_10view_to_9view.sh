# 1. Set videos
cd /mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/
vfiles=`find . -name \*.mp4 | grep -v com3d | grep -v sktdraw |grep -v keypoints | grep -v mask \
| grep -v -E '[0-9].mp4' | grep -v reverse  | grep -v crop | grep -v wtxwt_social`

# 2A. Crop and reverse videos
echo "$vfiles" | cat -n  \
| xargs -P 4 -l bash -c \
'ffmpeg -hide_banner -gpu $(($0%4)) -c:v hevc_cuvid \
-i $1  -filter_complex \
"crop=3840:2400:0:0, split=5[canvasORG][canvas1][canvas2][canvas3][canvas4], \
[canvas1]crop=1280:800:1280:0,hflip[flip1], \
[canvas2]crop=1280:800:0:800,hflip[flip2], \
[canvas3]crop=1280:800:2560:800,hflip[flip3], \
[canvas4]crop=1280:800:0:1600,hflip[flip4], \
[canvasORG][flip1]overlay=1280:0[canvasV1], \
[canvasV1][flip2]overlay=0:800[canvasV2], \
[canvasV2][flip3]overlay=2560:800[canvasV3], \
[canvasV3][flip4]overlay=0:1600" \
 -gpu $(($0%4)) -c:v hevc_nvenc -preset p6 -tune hq -b:v 30M \
 -y $1_crop9.mp4'

# 2B. Crop only
echo "$vfiles" | cat -n  \
| xargs -P 4 -l bash -c \
'ffmpeg -hide_banner -gpu $(($0%4)) -c:v hevc_cuvid \
-i $1  -filter_complex \
"crop=3840:2400:0:0" \
 -gpu $(($0%4)) -c:v hevc_nvenc -preset p6 -tune hq -b:v 30M \
 -y $1_crop9.mp4'

# 3. rename videos
echo "$vfiles" | sed  's/$/_crop9.mp4/' | xargs rename 's/.mp4_crop/_crop/' -d
