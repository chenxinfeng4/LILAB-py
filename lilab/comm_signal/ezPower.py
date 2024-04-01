import numpy as np
from scipy import signal, ndimage

def ezPower(Fs:float, data_1v:np.ndarray, doDB:bool=True, ifshow=True,
            doSmooth=True):
    data_1v = np.array(data_1v)
    assert data_1v.ndim == 1, "data_1v must be 1-D array"
    assert data_1v.size > 1024, "not enough data"

    frequencies, power_spectrum = signal.periodogram(data_1v, fs=Fs)
    frequencies, power_spectrum = frequencies[1:], power_spectrum[1:]
    Ns = len(power_spectrum)

    if doSmooth and Ns > 1024*4:
        window_size = Ns//(1024*4)
        p = ndimage.uniform_filter1d(power_spectrum, size=window_size)[::window_size]
        p = ndimage.gaussian_filter1d(p, sigma=2)
        f = frequencies[::window_size]
    else:
        p, f = power_spectrum, frequencies
        
    if doDB:
        ylabel = "Power Spectrum (dB)"
        p2 = 10 * np.log10(p)
    else:
        ylabel = "Power Spectrum"
        p2 = p

    if ifshow:
        import matplotlib.pyplot as plt
        plt.plot(f, p2)
        plt.ylabel(ylabel)
        plt.xlabel("Frequency (Hz)")

    return f, p2