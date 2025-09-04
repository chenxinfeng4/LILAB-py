import ffmpegcv
from ffmpegcv.ffmpeg_writer_noblock import FFmpegWriterNoblock
import tqdm
import time
vidin = ffmpegcv.VideoCapture('/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230724/2023-07-26_14-52-07CwxDb_3_sktdraw_smoothed_foot.mp4')
vidout = FFmpegWriterNoblock(
        ffmpegcv.VideoWriterStreamRT,
        "rtsp://10.50.60.6:8554/mystream_behaviorlabel_result"
    )

for frame in tqdm.tqdm(vidin):
    vidout.write(frame)
    time.sleep(0.03)
    