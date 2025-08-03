# python -m lilab.multiview_scripts.rat2d_kptvideo VIDEO.mp4 black.mat white.mat
# %%
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os.path as osp
import json
import cv2
import argparse
import mmcv
import tqdm

# %%
mat_file_black = "/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_30fps/whiteblack/15-37-42-white-copy/dlc/15-37-42_output_1-black.mat"
mat_file_white = "/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_30fps/whiteblack/15-37-42-white-copy/dlc/15-37-42_output_1-white.mat"
video = osp.join(osp.dirname(mat_file_black), "15-37-42_output_1.mp4")

# 定义个体颜色，rgb顺序
color_dict = {"black": (36, 173, 243), "white": (96, 30, 31), "dot": (255, 155, 54),
              0: (36, 173, 243), 1: (96, 30, 31), 2: (255, 155, 54),
              3: (87,20,102), 4: (178, 71, 13), 5: (133, 107, 24), 6: (87, 20, 102),
              7: (80, 111, 211)}
# %%
linkbody = np.array(
    [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [3, 6],
        [3, 8],
        [6, 7],
        [4, 6],
        [4, 8],
        [8, 9],
        [4, 10],
        [4, 12],
        [6, 10],
        [8, 12],
        [5, 10],
        [10, 11],
        [5, 12],
        [12, 13],
    ]
)
marker_types = [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1]


def cv_triangle(image, point_centor, size, color):
    r = size / 2
    r0_5, r_8 = 0.5 * r, 0.87 * r
    M = np.array([[r_8, -r0_5], [-r_8, -r0_5], [0, r]])
    triangle_corner = np.array(point_centor) + M
    cv2.fillPoly(image, [triangle_corner.astype(np.int32)], color=color)


def cv_plot_skeleton_aframe(aframe, point2d_aframe, name):
    # colors
    aframe = aframe.copy()
    # identitycolor = [97, 0, 15] if name=='white' else [0, 78, 130]  #RGB
    # identitycolor = [97, 0, 15] if name=='white' else [0, 78, 130]  #RGB
    # identitycolor = [96, 30, 31] if name=='white' else [36, 173, 243]  #RGB #[96, 30, 31] 白鼠,[235, 219, 119]金色黑鼠
    identitycolor = color_dict[name]
    colors = [
        [255, 0, 0],
        [255, 0, 255],
        [255, 0, 255],
        identitycolor,
        [255, 255, 0],
        [0, 255, 0],
    ] + [identitycolor] * 8
    colors = np.array(colors, dtype=np.uint8)[:, ::-1].tolist()  # BGR to RGB
    for marker_type, color, point2d in zip(marker_types, colors, point2d_aframe):
        if np.any(np.isnan(point2d)):
            continue
        point2d = tuple(point2d.astype(np.int32).tolist())
        if marker_type:
            cv_triangle(aframe, point2d, 16, color)
        else:
            cv2.circle(aframe, point2d, 6, color, -1)

    link2ds_from = point2d_aframe[linkbody[:, 0], :]
    link2ds_to = point2d_aframe[linkbody[:, 1], :]
    for link2d_from, link2d_to in zip(link2ds_from, link2ds_to):
        if np.any(np.isnan(link2d_from)) or np.any(np.isnan(link2d_to)):
            continue
        link2d_from = tuple(link2d_from.astype(np.int32).tolist())
        link2d_to = tuple(link2d_to.astype(np.int32).tolist())
        cv2.line(aframe, link2d_from, link2d_to, identitycolor[::-1], 1, cv2.LINE_AA)

    return aframe


# %%
def plot_video(video, pts2d_black, pts2d_white):
    vid = mmcv.VideoReader(video)
    assert (
        len(vid) == len(pts2d_black) == len(pts2d_white)
    ), "vid and pts2d must have the same length"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_file = osp.splitext(video)[0] + "_sktdraw.mp4"
    vidout = cv2.VideoWriter(output_file, fourcc, vid.fps, (vid.width, vid.height))

    for iframe in tqdm.tqdm(range(len(vid))):
        frame, pts2d_b_now, pts2d_w_now = (
            vid[iframe],
            pts2d_black[iframe],
            pts2d_white[iframe],
        )
        if not np.all(np.isnan(pts2d_b_now)):
            frame = cv_plot_skeleton_aframe(frame, pts2d_b_now, name="black")
        if not np.all(np.isnan(pts2d_w_now)):
            frame = cv_plot_skeleton_aframe(frame, pts2d_w_now, name="white")
        vidout.write(frame)

    vidout.release()


def main(video, mat_file_black=None, mat_file_white=None):
    assert osp.exists(video), "video not found"
    assert (
        mat_file_black or mat_file_white
    ), "mat_file_black or mat_file_white must be provided"
    if mat_file_black and mat_file_white:
        pts2d_black = sio.loadmat(mat_file_black)["points_2d"]
        pts2d_white = sio.loadmat(mat_file_white)["points_2d"]
    elif mat_file_black:
        pts2d_black = sio.loadmat(mat_file_black)["points_2d"]
        pts2d_white = np.zeros_like(pts2d_black) + np.nan
    elif mat_file_white:
        pts2d_white = sio.loadmat(mat_file_white)["points_2d"]
        pts2d_black = np.zeros_like(pts2d_white) + np.nan

    plot_video(video, pts2d_black, pts2d_white)


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the skeleton of a video")
    parser.add_argument("video", type=str, default=video, help="video file")
    parser.add_argument(
        "mat_file_black",
        type=str,
        default=mat_file_black,
        help="mat file of black points",
    )
    parser.add_argument(
        "mat_file_white",
        type=str,
        default=mat_file_white,
        help="mat file of white points",
    )
    args = parser.parse_args()
    main(args.video, args.mat_file_black, args.mat_file_white)
