import numpy as np
from scipy import signal

def lowerBand(dataIn, fc:float, sampleFs:float, order:int=4):
    dataIn = np.array(dataIn)
    assert dataIn.ndim == 1, "dataIn must be 1D"
    assert fc < (sampleFs-1)/2, "bandFs must be in [0, sampleFs/2]"
    
    Wn = 2 * fc / sampleFs
    b, a = signal.butter(order, Wn, btype='low')
    signal_output = signal.filtfilt(b, a, dataIn)
    return signal_output
