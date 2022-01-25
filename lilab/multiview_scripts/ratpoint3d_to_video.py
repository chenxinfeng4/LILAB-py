import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.io
import matplotlib.animation as animation
from collections import defaultdict
import os.path as osp
import argparse


def init_3d_plot():
    fig = plt.figure(figsize=(8,6), dpi=100)  # 800x600 pixels
    fig.add_axes([0,0,1,1], projection='3d')
    ax = fig.get_axes()[0]
    ax.set_title('3D Posture')
    ax.grid()
    ax.tick_params(length=0)
    # set the xlim and ylim of the plot
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_zlim(-2, 30)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ## create a circal plane by meshgrid
    radius = 24
    X, Y = np.meshgrid(np.arange(-radius, radius, 2), np.arange(-radius, radius, 2))
    ind_in = np.sqrt(X**2 + Y**2) < radius
    Z = np.zeros_like(X) - 1.0
    Z[np.invert(ind_in)] = np.nan
    ax.plot3D(X.flatten(), Y.flatten(), Z.flatten(), color='#13beb8', linewidth=0.5)
    ax.plot3D(X.T.flatten(), Y.T.flatten(), Z.flatten(), color='#13beb8', linewidth=0.5)
    ax.azim = -124
    return fig, ax

hplot_dict = defaultdict(dict)
linkbody = np.array([[0,1],[0,2],[1,3],[2,3],[3,6],[3,8],[6,7],[4,6], [4,8],[8,9],
                     [4,10], [4,12],[6,10],[8,12],[5,10],[10,11], [5,12], [12,13]])


def plot_skeleton_aframe(point3d_aframe, name, createdumpy=False):
    hplots = hplot_dict[name]
    if createdumpy:
        identitycolor = '#004e82' if name=='white' else '#61000f'
        markersize = 6
        hplots['leftbody'] = plt.plot(np.nan,np.nan, linestyle = 'None', marker='o', markeredgecolor='none', markerfacecolor=identitycolor, markersize=markersize)[0]
        hplots['rightbody'] = plt.plot(np.nan,np.nan, linestyle = 'None', marker='v', markeredgecolor='none', markerfacecolor=identitycolor, markersize=markersize+2)[0]
        hplots['nose'] = plt.plot(np.nan,np.nan, linestyle = 'None', marker='o', markeredgecolor='none', markerfacecolor='r', markersize=markersize)[0]
        hplots['earL'] = plt.plot(np.nan,np.nan, linestyle = 'None', marker='o', markeredgecolor='none', markerfacecolor='#ff00ff', markersize=markersize)[0]
        hplots['earR'] = plt.plot(np.nan,np.nan, linestyle = 'None', marker='v', markeredgecolor='none', markerfacecolor='#ff00ff', markersize=markersize+2)[0]
        hplots['back'] = plt.plot(np.nan,np.nan, linestyle = 'None', marker='o', markeredgecolor='none', markerfacecolor='#ffff00', markersize=markersize)[0]
        hplots['tail'] = plt.plot(np.nan,np.nan, linestyle = 'None', marker='o', markeredgecolor='none', markerfacecolor='#00ff00', markersize=markersize)[0]
        hplots['lines'] = plt.plot(np.nan,np.nan, color=identitycolor, linewidth=1)[0]
    else:
        line3d_from = point3d_aframe[linkbody[:,0],:]
        line3d_to = point3d_aframe[linkbody[:,1],:]
        line3d_array = np.zeros((linkbody.shape[0]*3, 3))*np.nan
        line3d_array[::3,:] = line3d_from
        line3d_array[1::3,:] = line3d_to
        hplots['leftbody'].set_data_3d(*(point3d_aframe[[3,6,7,10,11],:].T))
        hplots['rightbody'].set_data_3d(*(point3d_aframe[[8,9,12,13],:].T))
        hplots['nose'].set_data_3d(*(point3d_aframe[0,:].T))
        hplots['earL'].set_data_3d(*(point3d_aframe[1,:].T))
        hplots['earR'].set_data_3d(*(point3d_aframe[2,:].T))
        hplots['back'].set_data_3d(*(point3d_aframe[4,:].T))
        hplots['tail'].set_data_3d(*(point3d_aframe[5,:].T))
        hplots['lines'].set_data_3d(*(line3d_array.T))
        plt.draw()

    return list(hplots.values())


def plot_skeleton_aframe_easy(point3d_aframe):
    init_3d_plot()
    plot_skeleton_aframe(None, name='white', createdumpy=True)
    plot_skeleton_aframe(point3d_aframe, name='white', createdumpy=False)


def animate(points_white, points_black, i):
    hplots_white = plot_skeleton_aframe(points_white[i], 'white', False)
    hplots_black = plot_skeleton_aframe(points_black[i], 'black', False)
    return hplots_white + hplots_black


def main_plot3d(matlab_white=None, matlab_black=None, outfile=None):
    # load & check matlab data
    if matlab_black and matlab_white:
        points_white = scipy.io.loadmat(matlab_white)['points_3d']
        points_black = scipy.io.loadmat(matlab_black)['points_3d']
    elif matlab_black and not matlab_white:
        points_black = scipy.io.loadmat(matlab_black)['points_3d']
        points_white = points_black * np.nan
    elif matlab_white and not matlab_black:
        points_white = scipy.io.loadmat(matlab_white)['points_3d']
        points_black = points_white * np.nan
    else:
        raise ValueError('None input')

    if not outfile:
        outfile = (matlab_white or matlab_black).replace('.mat', '.mp4')


    assert points_white.shape == points_black.shape, 'points_white and points_black should have the same shape'
    assert points_white.shape[1:] == (14, 3) , 'points should be 14parts x 3xyz'   # 14 points, 3d
    assert points_white.shape[0] > 5, 'points should be more than 5'

    print('Accepted samples:', points_white.shape[0])

    # init the figure
    fig, ax = init_3d_plot()
    plot_skeleton_aframe(None, 'white', True)
    plot_skeleton_aframe(None, 'black', True)

    # animate
    fps = 15
    writer = animation.writers['ffmpeg'](fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    ani = animation.FuncAnimation(fig,lambda i : animate(points_white, points_black, i), 
                                  len(points_white), interval=1)
    ani.save(outfile, writer=writer)
    return outfile


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Plot 3d skeleton')
    argparser.add_argument('--matlab_white', type=str, help='matlab file of white person')
    argparser.add_argument('--matlab_black', type=str, help='matlab file of black person')
    args = argparser.parse_args()
    main_plot3d(args.matlab_white, args.matlab_black)
    