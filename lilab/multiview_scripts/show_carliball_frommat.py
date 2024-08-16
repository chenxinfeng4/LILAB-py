# python -m lilab.multiview_scripts.show_carliball_frommat ../data/carliball/carliball_keypoints.mat
# %%
import pickle
import numpy as np
import ffmpegcv
import tqdm
import cv2
import argparse


matfile = "/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/ball/2022-04-29_17-58-45_ball_mat.pkl"


def load_mat(matfile):
    data = pickle.load(open(matfile, "rb"))

    # %%
    keypoint = data["keypoints"].copy()
    views_xywh = data["views_xywh"]

    for keypoint_view, crop_xywh in zip(keypoint, views_xywh):
        keypoint_view[:] += np.array(crop_xywh[:2] + [0])
    return keypoint, views_xywh, data


def keypoint_to_video(keypoint, data, gpu=0):
    # %%
    resize = (1440, 1200)

    vfile = data["info"]["vfile"]
    vin = ffmpegcv.VideoCaptureNV(vfile, resize=resize, resize_keepratio=False, gpu=gpu)
    assert len(vin) == keypoint.shape[1]
    orisize = (vin.origin_width, vin.origin_height)
    scale = (resize[0] / orisize[0], resize[1] / orisize[1])

    keypoint[..., 0] *= scale[0]
    keypoint[..., 1] *= scale[1]

    # %%
    vout = ffmpegcv.VideoWriterNV(
        vfile.replace(".mp4", "_keypoints.mp4"), fps=vin.fps, gpu=gpu
    )
    for i, frame in enumerate(tqdm.tqdm(vin)):

        keypoint_i = keypoint[:, i, ...]
        keypoint_i = keypoint_i.reshape(-1, 3)
        keypoint_i_xy = keypoint_i[keypoint_i[:, 2] > 0.73][:, :2]
        keypoint_i_xy = keypoint_i_xy.astype(np.int32)

        # draw keypoints
        for xy in keypoint_i_xy:
            cv2.circle(frame, tuple(xy), 5, (0, 0, 255), -1)
        frame = cv2.putText(
            frame, str(i), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2
        )
        vout.write(frame)

    vout.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("matfile", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    keypoint, views_xywh, data = load_mat(args.matfile)
    keypoint_to_video(keypoint, data, args.gpu)
