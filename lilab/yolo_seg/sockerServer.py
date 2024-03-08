from lilabnext.multview_marmoset_track import t1c_realtime_position_daemon_socket as t1c_socket
import json
import numpy as np
import threading

t1c_socket.PORT = 8092
p3d = np.zeros((2,14,3), dtype=float)
p2d = np.zeros((9,2,14,2), dtype=int)

def get_com3d():
    com3d_str = json.dumps(p3d.tolist())
    return com3d_str

def get_com2d():
    com2d_str = json.dumps(p2d.tolist())
    return com2d_str

t1c_socket.get_com3d = get_com3d
t1c_socket.get_com2d = get_com2d

def start_socketserver_background():
    threading.Thread(target=t1c_socket.serve_forever).start()
