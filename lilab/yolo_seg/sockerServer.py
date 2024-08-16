import picklerpc
import numpy as np
import threading
import time

port = 8092
server = picklerpc.PickleRPCServer(('0.0.0.0', port))
p3d = np.zeros((2,14,3), dtype=float)
p2d = np.zeros((9,2,14,2), dtype=int)
datadict = dict(p3d=p3d, p2d=p2d, label_str='')

class QueueNumpy:
    def __init__(self):
        self.queue = np.zeros((31,), dtype=float) + np.nan
        self.maxlen = len(self.queue)
        self.head = 0
        self.timenow = time.time()

    def flush(self):
        self.queue[:] = np.nan
        self.head = 0
        self.timenow = time.time()
    
    def put(self, x:float):
        self.queue[self.head] = x
        self.head = (self.head + 1) % self.maxlen
        self.timenow = time.time()

    def get(self) -> float:
        # if time.time() - self.timenow > 1:
        #     return np.nan
        res = np.nanmedian(self.queue)
        return res

bhv_queue = QueueNumpy()
usv_queue = QueueNumpy()

@server.register()
def bhv_queue_put(x:float) -> None:
    bhv_queue.put(x)


@server.register()
def usv_queue_put(x:float) -> None:
    usv_queue.put(x)


@server.register()
def bhv_queue_get() -> float:
    return bhv_queue.get()


@server.register()
def usv_queue_get() -> float:
    return usv_queue.get()


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


if __name__ == '__main__':
    start_socketserver_background()
