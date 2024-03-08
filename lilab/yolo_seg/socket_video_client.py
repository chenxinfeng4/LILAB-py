#%%
import socket
import numpy as np
import cv2
import json
from lilab.multiview_scripts.rat2d_kptvideo import cv_plot_skeleton_aframe
import time
import ffmpegcv
from ffmpegcv.ffmpeg_noblock import ReadLiveLast
# vid = ReadLiveLast(ffmpegcv.VideoCaptureStreamRT, 'rtmp://10.50.60.6:1935/mystream', crop_xywh=[1280,0,1280,800], pix_fmt='bgr24')
# vid = ReadLiveLast(ffmpegcv.VideoCaptureStreamRT, 'rtmp://10.50.60.6:1935/mystream', crop_xywh=[1280,0,1280,800], pix_fmt='bgr24')
# vid = ReadLiveLast(ffmpegcv.VideoCaptureStreamRT, 'rtmp://localhost:1935/mystream', crop_xywh=[1280,0,1280,800], pix_fmt='bgr24')
# vid = ReadLiveLast(ffmpegcv.VideoCaptureStreamRT, 'rtsp://10.50.5.83:8554/mystream', crop_xywh=[1280,0,1280,800], pix_fmt='bgr24')
vid = ReadLiveLast(ffmpegcv.VideoCaptureStreamRT, 'rtsp://10.50.60.6:8554/mystream', crop_xywh=[1280,0,1280,800], pix_fmt='bgr24')
# print(vid.ffmpeg_cmd)
ret, frame = vid.read()
vid.count = 400000

while True:
    ret, frame = vid.read()
    time.sleep(0.1)
    if ret: break

print(frame.shape)

tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serve_ip = '10.50.60.6'
serve_port = 8092
tcp_socket.connect((serve_ip, serve_port))

def send_read(send_data):
    send_data_byte = send_data.encode("utf-8")
    tcp_socket.send(send_data_byte)

    from_server_msg = tcp_socket.recv(4096*4).decode("utf-8")
    while '\n' not in from_server_msg[-5:]:
        from_server_msg = from_server_msg + tcp_socket.recv(4096*4).decode("utf-8")
    return np.array(json.loads(from_server_msg))

#%%
img_shape = (800, 1280, 3)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
t = time.time()
while True:
    t_next = t+1/20
    dt = t_next - time.time()
    if dt>0:
        time.sleep(dt)
    # com3d = send_read('com3d')
    com2d = send_read('com2d')
    pts2d_b_now, pts2d_w_now = com2d[1]
    # frame = np.zeros(img_shape, dtype=np.uint8)
    ret, frame = vid.read()
    frame = cv_plot_skeleton_aframe(frame, pts2d_b_now, name = 'black')
    frame = cv_plot_skeleton_aframe(frame, pts2d_w_now, name = 'white')
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
    
    