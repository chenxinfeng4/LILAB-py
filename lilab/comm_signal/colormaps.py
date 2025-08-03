from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import numpy as np


def yellowblue():
    colors = [(0, [0, 0.75, 1]), 
              (0.5, [0, 0, 0]), (1, [1, 0.8, 0.3])]
    cmap = LinearSegmentedColormap.from_list('my_cmap', colors)
    return cmap


def redgrayblue():
    colors = [(0, [0, 0, 1]), (0.45, [0.49, 0.49, 0.49]),
              (0.55, [0.49, 0.49, 0.49]), (1, [1, 0, 0])]
    cmap = LinearSegmentedColormap.from_list('my_cmap', colors)
    cmap
    return cmap


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def get_cmap_new():
    cmap = cm.get_cmap('bwr')
    scalar_values = np.linspace(0, 1, 20)
    colors = cmap(scalar_values)
    colors[:6] = [colors[6]]
    colors[:10,:3] = rgb2gray(colors[:10,:3])[:,None]
    cmap_new = cm.colors.LinearSegmentedColormap.from_list('my_colormap', colors)
    return cmap_new

def get_cmap_dark_hsv():
    colors = [(0, '#005A97'), (0.15, '#005A97'),
                (0.35, '#006401'), (0.45, '#006401'), 
                (0.65, '#b26400'), (0.75, '#b26400'), 
                (0.9, '#B00001'), (1, '#B00001')]
    return LinearSegmentedColormap.from_list('my_cmap', colors)

def get_cmap_reds_blues_hsv():
    colors_r = [(0.0, '#FFFFFF'), (0.2, '#FFFFFF'), (0.60, '#c37574'),
                (0.86, '#881826'), (1, '#321e1d')]
    colors_b = [(0.0, '#FFFFFF'), (0.2, '#FFFFFF'), (0.60, '#5f8b97'),
                (0.86, '#18605a'), (1, '#040807')]
    
    return [LinearSegmentedColormap.from_list('my_cmap', colors_r),
            LinearSegmentedColormap.from_list('my_cmap', colors_b)]
