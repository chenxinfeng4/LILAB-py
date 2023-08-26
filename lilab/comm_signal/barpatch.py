import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings


def barpatch(x:list, x_dur:list, HL:list=None, color='red', **kwargs):
    """
    绘出类似 bar 的 patch 图

    Parameters:
    x : array_like
        bar(x), 1_vector.
    x_dur : array_like
            duration of x, 1_vector.
    HL : tuple
         High and Low, 2_nums.
    color : str
            color of the patch graph.
    kwargs : pars for 'plot()'.

    Returns:
    --------
    hpatch : handle for plot-line, 1x1.
    """
    # 准备参数 与 barline 一致
    if HL is None:
        HL = [0, 1]
    elif type(HL) in [int, float]:
        HL = [HL, HL + 1]
    elif len(HL) != 2:
        raise ValueError("Entry a number")
    
    HL = np.array(HL)
    x = np.array(x)
    x_dur = np.array(x_dur)
    if x.size == 0:
        warnings.warn("barpatch for empty data")
        return []
    assert x.ndim == 1, "x must be 1-D array"
    assert x_dur.ndim == 1, "x_dur must be 1-D array"
    
    # 制作数据
    xy_l = [(x0, HL[0]) for x0 in x]
    w_l = x_dur
    h_l = (HL[1] - HL[0]) * np.ones_like(x)

    # 画出数据
    hpatch_l = [Rectangle(xy, w, h, color=color, alpha=0.4, linewidth=0, **kwargs)
                for (xy, w, h) in zip(xy_l, w_l, h_l)]
    
    ax = plt.gca()
    for hpatch in hpatch_l: ax.add_patch(hpatch)

    plt.xlim([0,10])
    return hpatch_l
