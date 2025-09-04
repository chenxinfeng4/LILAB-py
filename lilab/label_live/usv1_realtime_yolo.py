# conda activate mmdet
import os
import cv2
import numpy as np
import ffmpegcv
from ffmpegcv.ffmpeg_writer_noblock import FFmpegWriterNoblock
from lilab.timecode_tag.decoder import getDecoder
from lilab.timecode_tag.netcoder import Netcoder
import picklerpc
from ultralytics import YOLO
import itertools
import tqdm
import torch
from lilab.label_live.water_timecode import water_timecode


checkpoint = "/home/liying_lab/chenxinfeng/DATA/ultralytics/work_dirs/yolov8n_det_640_usv/weights/last.pt"


def get_vidin():
    vid = ffmpegcv.ReadLiveLast(
        ffmpegcv.VideoCaptureStreamRT,
        "rtsp://10.50.60.6:8554/mystream_usv",
        pix_fmt="gray",
    )
    vidout = ffmpegcv.VideoWriterStreamRT("rtsp://10.50.60.6:8554/mystream_usv_preview")
    return vid, vidout


def main():
    vid, vidout = get_vidin()
    timecode_decoder = getDecoder()
    rpcclient = picklerpc.PickleRPCClient(("localhost", 8092))
    model = YOLO(checkpoint)  # load YOLO model
    model.to("cuda")
    model(np.zeros((vid.height, vid.width, 3), dtype=np.uint8), verbose=False)  # warmup
    print("YOLO model loaded")
    im0 = np.zeros((vid.height, vid.width), dtype=np.uint8)
    idx_line0 = 0
    nettimecoder = Netcoder()

    count_range = itertools.count()
    iter_process = tqdm.tqdm(count_range, desc="worker[{}]".format(3), position=int(3))
    frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    for idx in iter_process:
        ret, gray = vid.read()
        if not ret:
            break
        im = gray = gray.squeeze()
        timecode, *_ = timecode_decoder(gray)
        dt1 = nettimecoder.getTimeDelay(timecode)
        rpcclient.delay_usv_in(dt1)
        idx_line = np.argmax(np.mean(255 - gray[80:-40, 29:-11], 0))
        if idx_line0 - idx_line < 600 or idx_line0 < 600:
            idx_line0 = idx_line
            im0 = im[80:-40, 29:-11].copy()
            # water_timecode(frame_bgr, timecode)
            # vidout.write(frame_bgr)
            # continue

        idx_line0 = idx_line
        im0 = cv2.resize(im0, (640, 640))
        image_tensor = torch.from_numpy(im0[None][[0, 0, 0]]).float()[None] / 255.0
        image_tensor = image_tensor.to("cuda")
        with torch.no_grad():
            result = model(image_tensor, verbose=False, conf=0.02, iou=0.5)[0]

        boxes = result.boxes
        frame_bgr = cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)

        xyxy_l = boxes.xyxy.cpu().numpy().astype(np.int32)
        xyxy_l[:, 1] = 50
        xyxy_l[:, 3] = im0.shape[0] - 50
        cls_l = boxes.cls.cpu().numpy().astype(np.int32)
        rpcclient.label_usv_int(cls_l)
        for i, (x1, y1, x2, y2) in enumerate(xyxy_l):
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame_bgr,
                str(cls_l[i]),
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        # resize the frame_bgr to wh=640,480
        frame_bgr = cv2.resize(frame_bgr, (640, 480))
        water_timecode(frame_bgr, timecode)
        vidout.write(frame_bgr)
        dt2 = nettimecoder.getTimeDelay(timecode)
        rpcclient.usv_queue_put(dt2)
        rpcclient.delay_usv_out(dt2)

        dt_str = str(int(dt2)) if not np.isnan(dt2) else "x"
        iter_process.set_description("[USV Cls] delay={:>3}".format(dt_str))
    vid.release()
    vidout.release()
