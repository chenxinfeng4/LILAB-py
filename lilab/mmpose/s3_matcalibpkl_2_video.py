# python -m lilab.mmpose.s3_matcalibpkl_2_video /xxx/xxx.matcalibpkl
import os.path as osp
import pickle
import argparse
from lilab.multiview_scripts.ratpoint3d_to_video import plot_skeleton_aframe, animation, animate, init_3d_plot_350 as init_3d_plot


def plot_video(pts3d_black, pts3d_white, outfile, fps=30):
    fig, ax = init_3d_plot()
    plot_skeleton_aframe(None, 'white', True)
    plot_skeleton_aframe(None, 'black', True)
    writer = animation.writers['ffmpeg'](fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    ani = animation.FuncAnimation(fig,lambda i : animate(pts3d_black, pts3d_white, i), 
                                  len(pts3d_black), interval=1)
    ani.save(outfile, writer=writer)
    return outfile


def main(pkl):
    outfile = osp.splitext(pkl)[0] + '_3d.mp4'
    pkldata = pickle.load(open(pkl, 'rb'))
    fps = pkldata['info']['fps']
    pts3d = pkldata['keypoints_xyz_baglobal']
    pts3d_black = pts3d[:, 0, :, :]
    pts3d_white = pts3d[:, 1, :, :]
    plot_video(pts3d_black, pts3d_white, outfile, fps)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Plot 3d skeleton')
    argparser.add_argument('pkl', type=str, help='matcalibpkl file of keypoints')
    args = argparser.parse_args()
    main(args.pkl)
