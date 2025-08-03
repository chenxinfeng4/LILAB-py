# from lilab.comm_signal.BF_plotRaster import BF_plotRasterCell
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from .line_scale import line_scale

def BF_plotRasterCell(dat_cell, color='k', linebg=0, linehigh=0.8, show_hist=True, ax=None):
    """
    用cell格式储存光栅数据

    :param dat_cell: 光栅数据，1D 数组 of 数值 (cell_1v of 数值)
    :param color_3arr: 光栅颜色 (可选) (1D 数组 of 3 数字)
    :param linebg: 光栅起始点 (可选) (数字)
    :param linehigh: 光栅高度 (可选) (数字)

    :return h: 光栅的 line 句柄
    """

    # plot
    if ax is None:
        fig, ax = plt.subplots()
    h = []
    xtick_l, ytick_l = [], []
    for i, x in enumerate(dat_cell):
        if len(x) == 0: continue
        xtick = np.concatenate([[x0, x0, np.nan, np.nan] for x0 in x])
        ytick = np.concatenate([[linebg, linehigh, np.nan, np.nan] for x0 in x]) + i - 1
        xtick_l.append(xtick)
        ytick_l.append(ytick)
    
    xtick = np.concatenate(xtick_l)
    ytick = np.concatenate(ytick_l)
    h, = ax.plot(xtick, ytick, color=color, linewidth=1.5)
    if show_hist: plot_hist(dat_cell, ax=ax)
    return h


def plot_hist(data_cell, ax=None):
    n = len(data_cell)
    data_flat = np.concatenate(data_cell)
    c, e = np.histogram(data_flat, bins=50)
    smooth_data = savgol_filter(c, 13, 2)
    median = np.median(smooth_data)
    divation = np.max(np.abs(smooth_data - median))
    smooth_scaled = line_scale([median, median+divation], [n/2, n-1], smooth_data)
    if ax is None: ax = plt.gca()
    ax.plot(e[:-1], smooth_scaled, 'blue')
