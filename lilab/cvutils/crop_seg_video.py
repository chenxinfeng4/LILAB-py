import os
import os.path as osp
import argparse
import random
import tqdm
import numpy as np
import cv2
import mmcv


def find_boundary(img):
    imgb = img[:, :, 0] > 0
    # find the range of the image
    xmin, xmax = imgb.any(0).nonzero()[0][[0, -1]]
    ymin, ymax = imgb.any(1).nonzero()[0][[0, -1]]
    return xmin, xmax, ymin, ymax


def auto_find_size(videopath):
    vid = mmcv.VideoReader(videopath)
    nframe = len(vid)
    assert nframe > 10, "video is too short"
    nchoose = min(nframe, 100)
    ichoose = sorted(random.sample(range(nframe), nchoose))
    xranges, yranges = [], []
    for i in ichoose:
        img = vid[i]
        xmin, xmax, ymin, ymax = find_boundary(img)
        xrange = xmax - xmin
        yrange = ymax - ymin
        xranges.append(xrange)
        yranges.append(yrange)

    margin = 0.4
    xlen = int(max(xranges) * (1 + margin))
    ylen = int(max(yranges) * (1 + margin))
    xylen = max(xlen, ylen)
    xylen = int(xylen / 2) * 2  # even the number of xylen
    return xylen, xylen


def main(videopath, xylen=None):
    if not xylen:
        xylen, _ = auto_find_size(videopath)
    vid = mmcv.VideoReader(videopath)
    outfile = osp.splitext(videopath)[0] + "_crop.mp4"
    vidout = cv2.VideoWriter(
        outfile, cv2.VideoWriter_fourcc(*"mp4v"), vid.fps, (xylen, xylen)
    )
    for img in tqdm.tqdm(vid):
        imgb = img[:, :, 0] > 0
        # find the center of imgb
        xcen = np.sum(np.sum(imgb, axis=0) / np.sum(imgb) * np.arange(imgb.shape[1]))
        ycen = np.sum(np.sum(imgb, axis=1) / np.sum(imgb) * np.arange(imgb.shape[0]))
        if np.isnan(xcen) or np.isnan(ycen):
            xcen = ycen = xylen / 2 + 1
        xmin, xmax = int(xcen - xylen / 2), int(xcen + xylen / 2) - 1
        ymin, ymax = int(ycen - xylen / 2), int(ycen + xylen / 2) - 1
        bboxes = np.array([xmin, ymin, xmax, ymax])
        imgout = mmcv.imcrop(img, bboxes=bboxes, pad_fill=0)
        vidout.write(imgout)

    vidout.release()
    return outfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="auto crop seged video")
    parser.add_argument("videopath", type=str, default="", help="path to video")
    parser.add_argument(
        "--xylen", type=int, default=None, help="height/width of the cropped video"
    )
    args = parser.parse_args()
    main(args.videopath, args.xylen)
