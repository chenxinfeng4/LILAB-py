#%%
from ffmpegcv.ffmpeg_reader_tcpfifo import VideoCaptureTCPFIFO
import numpy as np
import tqdm
import time

camsize_wh=(1280,800)
fifo_url = '/home/liying_lab/chenxinfeng/streampipe'

vid = VideoCaptureTCPFIFO(fifo_url, camsize_wh, pix_fmt='gray')
print(vid.ffmpeg_cmd)
# ret = False
# tbar = tqdm.tqdm()
# while not ret:
#     ret, frame = vid.read_()
#     time.sleep(0.01)
#     tbar.update()

# assert ret, 'Failed to read from TCPFIFO'
# assert not np.all(frame == 0), 'Got a non-black frame'

tbar = tqdm.tqdm()
ret = True
while ret:
    ret, frame = vid.read_()
    tbar.update()
print('done')