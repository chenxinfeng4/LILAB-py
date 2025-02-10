# from lilab.comm_signal.BF_plotwSEM import BF_plotwSEM
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import matplotlib.lines
import matplotlib.patches

def BF_plotwSEM(x, y, yerr, color=None) -> Union[matplotlib.lines.Line2D, matplotlib.patches.Polygon]:
    pass
    """
    Plots the data with the standard error of the mean as a shaded area.

    Input parameters:
    x     : The x-axis data
    y     : The y-axis data
    yerr  : The error in the y-axis data

    Output parameters:
    h       : The line plot
    h_patch : The shaded area plot
    """
    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)
    assert x.ndim == y.ndim == yerr.ndim == 1, "x, y, yerr must be 1-D array"
    assert x.size == y.size == yerr.size, "x, y, yerr must be same size"

    xerror = np.concatenate([x, np.flip(x)])
    yerror = np.concatenate([(y + yerr), np.flip(y - yerr)])

    h_patch = plt.fill(xerror, yerror, 'y', alpha=0.4, color=color)[0]
    h = plt.plot(x, y, color=h_patch.get_facecolor())[0]

    return h, h_patch
