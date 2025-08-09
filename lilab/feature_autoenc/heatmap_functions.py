import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from matplotlib.colors import LinearSegmentedColormap
import re
import pandas as pd

p = re.compile("startFrameReal([0-9]+)")
panimals = re.compile(r"\D{2,}")

sigma = 1.5
percentile = 30
n_bins = [50, 50]

class HeatmapType:
    def __init__(self, all_map_density, xe, ye):
        self.all_map_density = all_map_density
        self.xe = xe
        self.ye = ye
        self.local_maxes = None
        self.markers = None
        self.water_line_mask = None
        self.cluster_area = None
        self.n_peaks = None
        self.all_map_density_line = None
        self.percentile = 30

    def calculate(self):
        density_cutoff = np.percentile(self.all_map_density, self.percentile)
        density_mask = self.all_map_density > density_cutoff
        if False:
            local_maxes = peak_local_max(self.all_map_density, indices=False)
        else:
            peaks = peak_local_max(self.all_map_density)  # 默认返回坐标
            local_maxes = np.zeros_like(self.all_map_density, dtype=bool)
            local_maxes[tuple(peaks.T)] = True
        local_maxes[np.logical_not(density_mask)] = False  # remove noise
        self.markers, self.n_peaks = scipy.ndimage.label(
            local_maxes)  # markers 0-n_peaks

        pos_peak = dict()
        for i in range(self.n_peaks):
            id = i+1
            x = np.where(self.markers == id)[0][0]
            y = np.where(self.markers == id)[1][0]
            pos_peak[id] = (x, y)
        self.pos_peak = pos_peak

        self.water_line_mask = watershed(-self.all_map_density,
                                         self.markers, watershed_line=True) == 0
        self.cluster_area = watershed(-self.all_map_density,
                                      self.markers, watershed_line=False)
        all_map_density_line = self.all_map_density.copy()
        all_map_density_line[self.water_line_mask] = np.nan
        self.all_map_density_line = all_map_density_line

    def df_append_tsne(self, df, tsne1_2):
        df[['embeded_tsne1', 'embeded_tsne2']] = tsne1_2
        df["ind_tsne1"], df["ind_tsne2"], df['labels'] = self.embedding_to_label(tsne1_2)

    def scatter_remap(self, x, y, clip=False):
        if clip:
            x = np.clip(x, self.xe[0]+0.001, self.xe[-1]-0.001)
            y = np.clip(y, self.ye[0]+0.001, self.ye[-1]-0.001)
        x2 = (x-self.xe[0])/(self.xe[1]-self.xe[0])
        y2 = (y-self.ye[0])/(self.ye[1]-self.ye[0])
        return x2, y2

    def embedding_to_label(self, embedding_xy):
        labeled_map = self.cluster_area
        assert labeled_map.min() == 1
        ind_x, ind_y = self.scatter_remap(
            embedding_xy[:, 0], embedding_xy[:, 1], clip=True)
        ind_x, ind_y = np.floor(ind_x).astype(int), np.floor(ind_y).astype(int)
        label = labeled_map[ind_x, ind_y]
        return ind_x, ind_y, label

    def plt_heatmap(self, name='all_map_density'):
        # htobj.plt_heatmap()
        # htobj.plt_heatmap('custer_area')
        colorarray = plt.cm.rainbow(np.linspace(0, 1, 10))
        newcolorl = [[1, 1, 1, 1], *colorarray, [0.59, 0.0, 0.00, 1]]
        # cmap = LinearSegmentedColormap.from_list("Custom", newcolorl, N=55)
        cmap = self.cmap if hasattr(self, 'cmap') else 'rainbow'
        array = getattr(self, name)
        assert array.ndim == 2
        plt.imshow(array.T, cmap=cmap, origin='lower')

    def plt_boundary(self):
        x, y = np.where(self.water_line_mask)
        plt.scatter(x, y, marker='.', color='w', edgecolor='w', s=40)

    def plt_textwhitedot(self):
        for i, (x, y) in self.pos_peak.items():
            plt.scatter([x], [y], marker='o', color='w', edgecolor='w', s=80)

    def plt_text(self):
        for i, (x, y) in self.pos_peak.items():
            plt.text(x, y, int(i), size=20)

    def set_mask_by_densitymap(self, thr=0.3):
        densitymap = self.all_map_density
        percentile = thr*100
        density_cutoff = np.percentile(densitymap, percentile)
        density_mask = densitymap > density_cutoff
        mask = np.zeros_like(densitymap)
        masked = np.ma.masked_where(density_mask, mask)
        self.density_mask = density_mask
        self.masked_map = masked

    def plt_maskheatmap(self):
        if not hasattr(self, 'masked_map'):
            self.set_mask_by_densitymap()
        cmap = LinearSegmentedColormap.from_list("Custom", ['w', 'w'], N=3)
        plt.imshow(self.masked_map.T, cmap=cmap,
                   interpolation='none', origin='lower')

    def plt_clusterarea_values(self, values, vmax=None):
        if isinstance(values, pd.DataFrame):
            labels = values.index.values
            data = np.squeeze(values.values)
        elif isinstance(values, list) or isinstance(values, np.ndarray):
            labels = np.arange(len(values))+1
            data = np.array(values)
        elif isinstance(values, dict):
            labels = np.array(values.keys())
            data = np.array([values[k] for k in labels])
        else:
            raise 'input data not fit'
        assert len(labels) == self.n_peaks
        canvas = np.ones_like(self.cluster_area) * np.nan
        for label, v in zip(labels, data):
            canvas[self.cluster_area == label] = v
        plt.imshow(canvas.T, cmap='rainbow', vmax=vmax,
                   interpolation='none', origin='lower')


def heatmap(data, axlims=None, bins=n_bins, normed=True, sigma=sigma):
    """Generates gaussian-filtered 2D histogram of 2D data

    Args:
        data: Nx2 numerical array
        axlims: The leftmost and rightmost edges of the bins along each 
                dimension (if not specified explicitly in the bins parameters): 
                [[xmin, xmax], [ymin, ymax]]. 
                All values outside of this range will be considered outliers 
                and not tallied in the histogram.
        bins:   If int, the number of bins for the two dimensions (nx=ny=bins).
                If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
                If [int, int], the number of bins in each dimension (nx, ny = bins).
                If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
                A combination [int, array] or [array, int], where int is the number of bins and 
                array is the bin edges.
        normed: Default true, returns the probability density function at the bin, 
                bin_count / sample_count / bin_area.
                If false, returns count per bin.
        sigma:  scalar or sequence of scalers, standard deviation(s) for Gaussian kernel.
                The standard deviations of the Gaussian filter are given for each axis as a 
                sequence, or as a single number, in which case it is equal for all axes.



    Returns: 
        2D array of filtered histogram, 
        1D array of bin edges in dim 0, 
        1D array of bin edges in dim 1
    
    """
    from scipy.ndimage.filters import gaussian_filter

    if axlims is None:
        xmin, xmax = data[:,0].min(), data[:,0].max()
        ymin, ymax = data[:,1].min(), data[:,1].max()
        xrange, yrange = xmax - xmin, ymax-ymin
        xedgelen, yedgelen = xrange / n_bins[0] * 2, yrange / n_bins[1] * 2
        xmin_re, xmax_re = xmin - xedgelen, xmax + xedgelen
        ymin_re, ymax_re = ymin - yedgelen, ymax + yedgelen
        axlims=[[xmin_re, xmax_re],[ymin_re, ymax_re]]

    # Initial histogram
    heatmap, xedges, yedges = np.histogram2d(data[:,0], data[:, 1],
            bins=bins, range=axlims, density=normed)
    # Convolve with Gaussian
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    return heatmap,xedges,yedges


