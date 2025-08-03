# from lilab.comm_signal.linehintTTLbin import linehintTTLbin
import matplotlib.pyplot as plt
import numpy as np


def linehintTTLbin(xticks, ttlbin, ylevel=1, color='r', linewidth=1):
    ttlbin_float = np.array(ttlbin, dtype=float) * ylevel
    ttlbin_float[ttlbin==0] = np.nan
    plt.step(xticks-(xticks[1]-xticks[0])/2, ttlbin_float, color=color, linewidth=linewidth)
    