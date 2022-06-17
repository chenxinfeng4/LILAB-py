# import lilab.cvutils.map_multiprocess_cuda as mmap_cuda
import multiprocessing as mp
from multiprocessing import Manager
import time
import random
import tqdm
import time
import torch
import os
import signal
from itertools import repeat
import psutil

ctx = mp.get_context('spawn')
queue_cuda = Manager().Queue()       #only used once
queue_workerpool = Manager().Queue() #hot get, hot put, the id
_workerpool = []                     #only used once


__all__ = ['workerpool_init', 'workerpool_compute_map', 'Worker']


class Worker:
    def __init__(self):
        self.id, self.cuda = queue_cuda.get()

    def compute(self, arg):
        cuda = self.cuda

    def compute_proxy(self, *arg, **args):
        self.compute(*arg, **args)
        process = mp.current_process()
        os.kill(process.pid, signal.SIGKILL)


class _MyWorker(Worker):
    def __init__(self, something = 'something'):
        super().__init__()
        time.sleep(random.random())
        self.somthing = None
        print(f'worker: {self.id}, cuda:{self.cuda}')

    def compute(self, video):
        time.sleep(random.random())
        time.sleep(2)
        print(video)


def workerpool_init(gpulist, WorkerClass, *args, **kwargs):
    queue_cuda.empty()
    queue_workerpool.empty()
    _workerpool.clear()
    for id, cuda in enumerate(gpulist):
        cuda = cuda % torch.cuda.device_count()
        queue_cuda.put((id, cuda))
        worker = WorkerClass(*args, **kwargs)    # create worker instances
        _workerpool.append(worker)
        queue_workerpool.put(id)

    print('Worker pool has {}'.format(queue_workerpool.qsize()))


def _workerpool_compute(arg, queue_workerpool, _workerpool):
    worker = queue_workerpool.get()
    _workerpool[worker].compute(arg)             #  worker instance do computing
    queue_workerpool.put(worker)
    process = mp.current_process()
    # print('worker {} kill {}'.format(worker, process.pid))
    pid = process.pid
    print('sub pid:', pid)
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
        print('killed sub child', child.pid)
    # os.kill(process.pid, signal.SIGKILL)


def workerpool_compute_map(iterable, use_cuda=True):
    # if use_cuda:
    #     pool = ctx.Pool(processes=queue_workerpool.qsize())
    # else:
    #     pool = mp.Pool(processes=queue_workerpool.qsize(),
    #                    initargs=(tqdm.tqdm.get_lock(),),
    #                    initializer=tqdm.tqdm.set_lock)
    
    # for arg in iterable:
    #     # 保留 queue_workerpool, _workerpool 的引用，否则多进程回报错
    #     time.sleep(0.2)
    #     pool.apply_async(_workerpool_compute, args=(arg, queue_workerpool, _workerpool))
    with ctx.Pool(processes=queue_workerpool.qsize()) as pool:
        pool.starmap(_workerpool_compute, zip(iterable, repeat(queue_workerpool), repeat(_workerpool)))

    pool.close()
    pool.join()

    print('End of pool.close()')


if __name__ == '__main__':
    workerpool_init(['0', '1', '2','0','1','2'], _MyWorker)
    workerpool_compute_map(range(20))
    