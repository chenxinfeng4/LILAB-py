import numpy as np

def cutwave(wave, Fs, timerg):
    """
    Reads the sampled data and returns a cut data in the time range specified, 
    while the insufficient data is padded with NaN.

    Input parameters:
    wave   : The sampled data
    Fs     : The sampling rate
    timerg : The time range

    Output parameters:
    timepick : The time axis
    wavepick : The cut wave, padded with NaN if needed
    """
    if isinstance(timerg, (int, float)):
        timerg = np.array([0, timerg])
    timerg = np.array(timerg)
    assert timerg.ndim == 1 and timerg.size == 2, 'timerg must be 1-D array!'
    tickrg = timerg*Fs
    tickrg = np.floor(tickrg).astype(int)
    tickall = np.arange(tickrg[0], tickrg[1])
    timepick = tickall / Fs
    tickmax = len(wave)
    wavepick = np.nan*np.ones_like(tickall, dtype=float)

    # index indicators
    ind_inside = (tickall >= 0) & (tickall < tickmax)
    wavepick[ind_inside] = wave[tickall[ind_inside]]
    return timepick, wavepick
