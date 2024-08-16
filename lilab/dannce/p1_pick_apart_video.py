# python -m lilab.dannce.p1_pick_apart_video xxx/xxx/
# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
import ffmpegcv
import os
import os.path as osp
import tqdm
import cv2
import argparse

pklfile = "/mnt/ftp.rat/multiview_9/SHANK3HETxWT/2022-10-10/2022-10-10_15-18-10-MbHExWwt.smoothed_foot.matcalibpkl"


def pick_apart_video(pklfile):
    vfile = osp.join(osp.dirname(pklfile), osp.basename(pklfile).split(".")[0] + ".mp4")
    print(vfile)
    with open(pklfile, "rb") as f:
        matcalib = pickle.load(f)

    kpt_3d = matcalib["keypoints_xyz_ba"]  # mm (T, nclass, K, 3)
    T = kpt_3d.shape[0]
    dist_list = np.zeros(T, dtype=np.float32)

    head_tail = np.linalg.norm(kpt_3d[:, :, 0] - kpt_3d[:, :, 5], axis=-1).flatten()
    head_tail_len = np.percentile(head_tail, 95)

    for iframe in range(T):
        full_dist = kpt_3d[iframe, 0][:, None, :] - kpt_3d[iframe, 1][None, :, :]
        dist = np.linalg.norm(full_dist, axis=-1)
        dist_list[iframe] = dist.min()

    thr = head_tail_len * 0.5
    line_thr = (dist_list > thr) * thr

    # plt.plot(dist_list)
    # plt.plot(line_thr)
    # plt.show()

    vfileout = osp.splitext(vfile)[0] + "_apart.mp4"
    with ffmpegcv.VideoCaptureNV(vfile, crop_xywh=[1280, 0, 1280, 800]) as vidin:
        with ffmpegcv.VideoWriterNV(vfileout, fps=vidin.fps) as vidout:
            for i, (frame, over_thr) in enumerate(zip(tqdm.tqdm(vidin), line_thr)):
                if over_thr:
                    frame = cv2.putText(
                        frame,
                        str(i),
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 0, 255),
                        2,
                    )
                    vidout.write(frame)


def main(video_folder):
    for pklfile in os.listdir(video_folder):
        if pklfile.endswith("smoothed_foot.matcalibpkl"):
            pick_apart_video(osp.join(video_folder, pklfile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_folder", type=str)
    args = parser.parse_args()

    main(args.video_folder)
