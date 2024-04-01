import picklerpc
import numpy as np
import threading


server = picklerpc.PickleRPCServer(('0.0.0.0', 8092))
p3d = np.zeros((2,14,3), dtype=float)
p2d = np.zeros((9,2,14,2), dtype=int)
datadict = dict(p3d=p3d, p2d=p2d)


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


def start_socketserver_background():
    threading.Thread(target=server.serve_forever).start()
