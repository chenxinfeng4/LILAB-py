# !/bin/bash
#bash /home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/cvutils/concat_video_bytime.sh
vfiles=`ls -d $PWD/*.mp4 | grep -v out.mp4  | sed -e 's/^/file /'`
echo "$vfiles" > .vfiles.txt
ffmpeg -f concat -safe 0 -i .vfiles.txt -c copy -y out.mp4

rm .vfiles.txt
