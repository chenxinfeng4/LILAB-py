import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def yellowblue_(m=16):
    # Blue-Black-Yellow
    clrs = np.array([[0, 0.75, 1], [0, 0, 0], [1, 0.8, 0.3]])
    y = np.array([-1, 0, 1])
    if m % 2:
        delta = min(1, 2/(m-1))
        half = (m-1)/2
        yi = delta * np.arange(-half, half+1)
    else:
        delta = min(1, 2/m)
        half = m/2
        yi = delta * np.arange(-half, half)
    cmap = np.interp(yi, y, clrs)
    return cmap


def yellowblue():
    colors = [(0, [0, 0.75, 1]), 
              (0.5, [0, 0, 0]), (1, [1, 0.8, 0.3])]
    cmap = LinearSegmentedColormap.from_list('my_cmap', colors)
    return cmap