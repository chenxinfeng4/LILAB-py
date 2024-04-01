nc -l 8082 > ~/streampipe

python /home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/yolo_seg/fifo_video_get.py
ffmpeg    -f flv -i /home/liying_lab/chenxinfeng/streampipe  -vf extractplanes=y -pix_fmt gray  -f rawvideo pipe: 1>/dev/null

vfile=/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/VPAxWT/DAY35/2023-01-01_15-54-20BwxDb.mp4
vfile=/home/liying_lab/chenxinfeng/testflv.flv
ffmpeg -re -i $vfile  -f flv -y ~/streampipe
