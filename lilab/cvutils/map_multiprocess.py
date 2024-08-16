# python -m lilab.cvutils.map_multiprocess .
import multiprocessing
from multiprocessing import Pool, Value
from tqdm import tqdm as _tqdm
import time, random

_iPool = Value("i", -1)

npool = 12


def _get_iPool():
    _iPool.value += 1
    return _iPool.value


def tqdm(iterable, **kwargs):
    position = _get_iPool()
    return _tqdm(iterable, position=position, desc=f"[{position+1}]", **kwargs)


def map(func, iterable):
    ncpu = multiprocessing.cpu_count()
    maxproc = min([npool, ncpu, len(iterable)])
    with Pool(
        processes=maxproc, initargs=(_tqdm.get_lock(),), initializer=_tqdm.set_lock
    ) as pool:
        poolReturn = pool.map(func, iterable)
    return poolReturn


def starmap(func, *args):
    ncpu = multiprocessing.cpu_count()
    maxproc = min([npool, ncpu, len(args[0])])
    with Pool(
        processes=maxproc, initargs=(_tqdm.get_lock(),), initializer=_tqdm.set_lock
    ) as pool:
        pool.starmap(func, *args)


def _testfun(arg):
    print("getting")
    time.sleep(random.random())
    print(arg)
    time.sleep(2)
    print("finish")


if __name__ == "__main__":
    map(_testfun, range(20))
