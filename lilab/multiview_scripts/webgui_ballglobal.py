from pywebio.input import *
from pywebio.output import *
from pywebio import pin
from pywebio import start_server
import numpy as np
import os
import os.path as osp
import glob
import shutil
import cv2
import mmcv
import PIL.Image
import lilab.cvutils.crop_videofast
import lilab.multiview_scripts.dlcBall_2_landmarks
import mmcv

"""
/home/liying_lab/chenxinfeng/deeplabcut-project/balllabel/
/home/liying_lab/chenxinfeng/DATA/multiview-project/tph2ko/ball.mp4
00:00:00
00:00:08
00:00:15
00:00:22
00:00:31
"""
subdirname = "ballglobal"
scope_msg = "ballglobal_messages"

kernel_open = np.ones((5, 5), np.uint8)
kernel_dilate = np.ones((200, 300), np.uint8)


def get_seconds_from_str(str):
    if ":" in str:
        h, m, s = str.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    else:
        return int(str)


def on_extract_and_crop_video():
    with use_scope(scope_msg, clear=True):
        names = ["t_global_ball" + str(i) for i in range(1, 6)]
        checkinputs = [pin.pin[name] for name in names]
        if not all(checkinputs):
            put_error("Please fill in all the time points!")
            raise Exception("Please fill in all the time points!")
            return
        put_text("Extracting images...")

        global_times = [get_seconds_from_str(pin.pin[name]) for name in names]
        if global_times != sorted(global_times):
            put_error("Please fill in the time points in ascending order!")
            raise Exception("Please fill in the time points in ascending order!")
            return

        ballvideopath = pin.pin.global_ball_path
        balldir = osp.dirname(ballvideopath)
        if not osp.exists(ballvideopath):
            put_error("The global ball video does not exist!")
            raise Exception("The global ball video does not exist!")
            return

        # use ffmpeg
        # ffmpeg -i <input> -filter:v "select='eq(t,3)-eq(t,5)-eq(t,14)'" <output>
        selecttimestr = "-".join(["eq(t,{})".format(t) for t in global_times])
        outdir = osp.join(balldir, subdirname)
        os.makedirs(outdir, exist_ok=True)
        shutil.copy(balldir + "/config_video.py", outdir)
        outballvideopath = osp.join(outdir, "ball_global.mp4")
        os.system(
            f'ffmpeg -i "{ballvideopath}" -t {global_times[-1]+1} -filter:v "select=\'{selecttimestr}\'" -vsync vfr -r 1 -y "{outballvideopath}"'
        )
        put_text("Extracting images done! Crop the images...")
        lilab.cvutils.crop_videofast.main(outdir)

        put_success("Done!")


def on_get_background_images():
    outdir = osp.join(osp.dirname(pin.pin.global_ball_path), subdirname)
    videos = sorted(glob.glob(outdir + "/*output*.mp4"))
    with use_scope(scope_msg, clear=True):
        put_text("Extracting background images...")
        for i, video in enumerate(videos):
            vid = mmcv.VideoReader(video)
            imgs = np.array([img for img in vid])
            imgbg = np.median(imgs, axis=0).astype(np.uint8)
            mmcv.imwrite(imgbg, osp.join(outdir, f"bgframe_view_{i+1}.png"))

            # show the foreground
            imgfg = imgbg.copy()
            for i, img in enumerate(imgs):
                mask = np.abs(img.astype(np.float32) - imgbg.astype(np.float32)) > 10
                mask = mask[:, :, 0].astype(np.uint8)
                # open geometry and dilate for the mask using cv2
                mask = cv2.morphologyEx(
                    mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_open
                )
                mask = cv2.dilate(mask, kernel_dilate)
                maskbin = mask.astype(bool)
                imgfg[maskbin, :] = img[maskbin, :]
            mmcv.imwrite(imgfg, osp.join(outdir, f"fgframe_view_{i+1}.png"))

        put_image_proxy = lambda i: put_image(
            open(osp.join(outdir, f"bgframe_view_{i+1}.png"), "rb").read()
        )
        put_table(
            [
                [put_image_proxy(0), put_image_proxy(1), put_image_proxy(2)],
                [put_image_proxy(3), put_image_proxy(4), put_image_proxy(5)],
            ]
        )
        put_success("Done!")


def on_deeplabcut_predict_ball():
    import deeplabcut as dlc

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    with use_scope(scope_msg, clear=True):
        videos_folder = osp.dirname(pin.pin.global_ball_path) + "/" + subdirname
        config_name = pin.pin.dlc_project_ball
        if osp.isdir(config_name):
            config_name = osp.join(config_name, "config.yaml")
        if not osp.isfile(config_name):
            put_error("The Deeplabcut project does not exist!")
            raise Exception("The Deeplabcut project does not exist!")
            return

        put_text("Processing:", videos_folder)
        videos = glob.glob(videos_folder + "/*output*.mp4")
        videos_labeled = glob.glob(videos_folder + "/*labeled.mp4")
        videos_orginal = list(set(videos) - set(videos_labeled))
        assert len(videos_orginal) == 6
        dlc.analyze_videos(config_name, videos=videos_orginal, save_as_csv=True)


def on_csv_to_landmarkjson():
    with use_scope(scope_msg, clear=True):
        csv_folder = osp.dirname(pin.pin.global_ball_path) + "/" + subdirname
        landmarkjson_path = osp.join(csv_folder, "landmarks_global_pixel.json")
        put_text("Converting csv to landmarkjson...")
        lilab.multiview_scripts.dlcBall_2_landmarks.main(csv_folder, landmarkjson_path)
        put_success("Done!")


def app(parent=None):
    if not parent:
        parent = "ballglobal"
        put_scope(parent)

    with use_scope(parent):
        pin.put_input("dlc_project_ball", label="The Deeplabcut project folder")
        pin.put_input("global_ball_path", label="The global path of ball")
        put_table(
            [
                ["Location", "1", "2", "3", "4", "5"],
                [
                    "Time",
                    pin.put_input("t_global_ball1", placeholder="00:00:01"),
                    pin.put_input("t_global_ball2", placeholder="00:00:02"),
                    pin.put_input("t_global_ball3", placeholder="00:00:03"),
                    pin.put_input("t_global_ball4", placeholder="00:00:04"),
                    pin.put_input("t_global_ball5", placeholder="00:00:05"),
                ],
            ]
        )
        put_buttons(
            [
                "1. extract and crop video",
                "1.5 get background images",
                "2. deeplabcut predict ball",
                "3. csv file to landmarkjson",
            ],
            onclick=[
                on_extract_and_crop_video,
                on_get_background_images,
                on_deeplabcut_predict_ball,
                on_csv_to_landmarkjson,
            ],
        )
        put_scope(scope_msg)
