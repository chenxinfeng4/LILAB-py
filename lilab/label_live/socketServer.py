import picklerpc
import numpy as np
import threading
import time
from typing import List

port = 8002
server = picklerpc.PickleRPCServer(('0.0.0.0', port))
p3d = np.zeros((2,14,3), dtype=float)
p2d = np.zeros((9,2,14,2), dtype=int)
datadict = dict(p3d=p3d, p2d=p2d, label_str='', 
                label_int=-1, label_usv_int=[],
                delay_usv_in=0, delay_usv_out=0,
                delay_bhv_in=0, delay_bhv_out=0,
                logfile='')

class QueueNumpy:
    def __init__(self):
        self.queue = np.zeros((5,), dtype=float) + np.nan
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

@server.register()
def label_int(arg=None):
    if arg is None:
        return datadict['label_int']
    else:
        datadict['label_int'] = arg


@server.register()
def label_usv_int(arg=None):
    if arg is None:
        return datadict['label_usv_int']
    else:
        datadict['label_usv_int'] = arg


@server.register()
def delay_usv_in(arg=None):
    if arg is None:
        return datadict['delay_usv_in']
    else:
        datadict['delay_usv_in'] = arg


@server.register()
def delay_usv_out(arg=None):
    if arg is None:
        return datadict['delay_usv_out']
    else:
        datadict['delay_usv_out'] = arg


@server.register()
def delay_bhv_in(arg=None):
    if arg is None:
        return datadict['delay_bhv_in']
    else:
        datadict['delay_bhv_in'] = arg


@server.register()
def delay_bhv_out(arg=None):
    if arg is None:
        return datadict['delay_bhv_out']
    else:
        datadict['delay_bhv_out'] = arg


@server.register()
def about():
    return 'Social-seq live server'

@server.register()
def logfile(arg=None):
    if arg is None:
        return datadict['logfile']
    else:
        datadict['logfile'] = arg


def start_socketserver_background():
    threading.Thread(target=server.serve_forever).start()


if __name__ == '__main__':
    start_socketserver_background()
