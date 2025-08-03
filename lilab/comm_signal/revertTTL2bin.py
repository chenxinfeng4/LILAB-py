# from lilab.comm_signal.revertTTL2bin import revertTTL2bin
import numpy as np

def revertTTL2bin(tRise:np.ndarray, tDur:np.ndarray, Fs=1, tlen=None) -> np.ndarray:
    assert len(tRise) == len(tDur)
    assert isinstance(tRise, np.ndarray) and isinstance(tDur, np.ndarray)
    if tlen is None:
        tlen = tRise.max() + tDur.max()
    
    data = np.zeros(np.ceil(tlen*Fs).astype(int), dtype=bool)
    tDown = tRise + tDur
    tRise_n_dt = np.ceil(tRise*Fs).astype(int)
    tDown_n_dt = np.ceil(tDown*Fs).astype(int)
    for tR, tD in zip(tRise_n_dt, tDown_n_dt):
        data[tR:tD] = True
    return data
