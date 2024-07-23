import picklerpc
import numpy as np
import threading

port = 8092
server = picklerpc.PickleRPCServer(('0.0.0.0', port))
p3d = np.zeros((2,14,3), dtype=float)
p2d = np.zeros((9,2,14,2), dtype=int)
datadict = dict(p3d=p3d, p2d=p2d, label_str='')


@server.register()
def set_p3d(p3d:np.ndarray) -> None:
    datadict['p3d'] = p3d


@server.register()
def set_p2d(p2d:np.ndarray) -> None:
    datadict['p2d'] = p2d


@server.register()
def get_p3d() -> np.ndarray:
    return datadict['p3d']


@server.register()
def get_p2d() -> np.ndarray:
    return datadict['p2d']

@server.register()
def label_str(arg=None):
    if arg is None:
        return datadict['label_str']
    else:
        datadict['label_str'] = arg

def start_socketserver_background():
    threading.Thread(target=server.serve_forever).start()
