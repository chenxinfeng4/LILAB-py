import numpy as np
from scipy import signal

def butterBand(dataIn, bandFs:list, sampleFs:float, order:int=2):
    dataIn = np.array(dataIn)
    bandFs = np.array(bandFs)
    assert dataIn.ndim == 1, "dataIn must be 1D"
    assert bandFs.shape == (2,), "bandFs must be 2 elements"
    assert bandFs[0]<bandFs[1]<(sampleFs-1)/2, "bandFs must be in [0, sampleFs/2]"
    
    Wn = 2 * bandFs / sampleFs
    b, a = signal.butter(order, Wn, btype='band')
    signal_output = signal.lfilter(b, a, dataIn)
    return signal_output
