import os
import os.path as osp
import argparse

vfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/2022-04-28_16-49-20_wwt_bwt_noiso.mp4'

dirname = osp.dirname(vfile)
tempname = osp.splitext(vfile)[0] + r'_%2dtime.mp4'
segment_time = '00:10:00'

cmd = f'ffmpeg -i "{vfile}" -c copy -segment_time {segment_time} -f segment -reset_timestamps 1 "{tempname}"'
os.system(cmd)
