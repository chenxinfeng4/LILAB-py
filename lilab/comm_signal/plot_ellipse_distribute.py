# from lilab.comm_signal.plot_ellipse_distribute import plt_ellipse
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def __plt_ellipse(x, y, ax, color):
    data = np.array([x, y])
    pca_ = PCA(n_components=2)
    pca_.fit(data.T)
    components = pca_.components_
    std_deviations = np.sqrt(pca_.explained_variance_)

    theta = np.linspace(0, 2*np.pi, 100)
    x = std_deviations[0] * np.cos(theta)
    y = std_deviations[1] * np.sin(theta)
    ellipse = np.vstack((x, y)).T.dot(components)*2 + np.mean(data, axis=1)

    ax.fill(ellipse[:, 0], ellipse[:, 1], edgecolor='#040404', facecolor=color, alpha=0.3)


def plt_ellipse(x, y, ax=None, color='red'):
    data = np.array([x, y])
    pca_ = PCA(n_components=2)
    pca_.fit(data.T)
    components = pca_.components_
    std_deviations = np.sqrt(pca_.explained_variance_)

    theta = np.linspace(0, 2*np.pi, 100)
    x = std_deviations[0] * np.cos(theta)
    y = std_deviations[1] * np.sin(theta)
    ellipse = np.vstack((x, y)).T.dot(components)*2 + np.mean(data, axis=1)

    if ax is None: ax=plt.gca()
    ax.fill(ellipse[:, 0], ellipse[:, 1], edgecolor='#040404', facecolor=color, alpha=0.3)
