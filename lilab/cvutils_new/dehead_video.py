# python -m lilab.cvutils_new.dehead_video VIDEO NUM_HEAD

import ffmpegcv
import subprocess
import re
import numpy as np
import tempfile
import os
import argparse
import time


def naughty_ffmpeg_copy(
    vinfo, split_pts_time_opt, split_k_frameinvideo, video, video_out_body
):
    # ffmpeg 无法精准定位到指定的 K-FRAME，令人十分难受，所以2阶段分步走。
    cmd_tail = f'ffmpeg -y -loglevel warning -ss {split_pts_time_opt:.3f}  -i "{video}"  -c:v copy "{video_out_body}" '
    os.system(cmd_tail)
    vinfo_body = ffmpegcv.video_info.get_info(video_out_body)
    if split_k_frameinvideo + vinfo_body.count == vinfo.count:
        return
    dcount = split_k_frameinvideo + vinfo_body.count - vinfo.count
    split_pts_time_opt_post = split_pts_time_opt + (dcount / vinfo.fps)

    # try to refine body video
    cmd_tail = f'ffmpeg -y -loglevel warning -ss {split_pts_time_opt_post:.3f} -i "{video}" -c:v copy "{video_out_body}" '
    os.system(cmd_tail)
    vinfo_body = ffmpegcv.video_info.get_info(video_out_body)
    assert split_k_frameinvideo + vinfo_body.count == vinfo.count


def dehead(video, num_dehead, ishevc=True):
    print('num_dehead:', num_dehead)
    video_concat = os.path.splitext(video)[0] + "_dehead.mp4"
    if num_dehead == 0:
        os.system(f"ffmpeg -loglevel warning -y -i {video} -c copy {video_concat}")
        return video_concat

    ext = '.hevc' if ishevc else '.h264'     #h264
    _, video_out_neck = tempfile.mkstemp(suffix=ext)
    _, video_out_body = tempfile.mkstemp(suffix=ext)
    K_frame_guess = 100 if ishevc else 250
    cmd = f"ffprobe -v quiet -select_streams v -show_entries packet=pts_time,flags -of compact -read_intervals 0%+#{num_dehead+K_frame_guess} {video} | cat -n | grep K_"
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    example_string = result.stdout
    assert len(example_string) > 0
    # pts_time_pattern = re.compile(r"pts_time=(.*?)\|")
    pts_time_pattern = re.compile(r"pts_time=(\d*\.\d+)")
    pts_times = pts_time_pattern.findall(example_string)
    pts_times_np = np.array([float(t) for t in pts_times])
    assert len(pts_times_np) > 0
    frame_pattern = re.compile(r" ([0-9]+)\tpacket")
    frame_numbers = frame_pattern.findall(example_string)
    frame_numbers_np = np.array([int(f) for f in frame_numbers]) - 1  # start from 0
    print('frame_numbers_np:', frame_numbers_np)
    assert len(frame_numbers_np) == len(pts_times_np)

    # 0---NUM_dehead----split_k_frameinvideo---video_end
    # video_out_neck: iframe = [NUM_dehead : split_k_frameinvideo]
    # video_out_body: iframe = [split_k_frameinvideo (time=split_pts_time_opt): end]
    ind_k_frame = np.where(frame_numbers_np > num_dehead)[0][0]
    split_pts_time = pts_times_np[ind_k_frame]
    split_pts_time_opt = split_pts_time - 0.1  # 开始截断的时间点 a bit earlier
    # split_pts_time_opt = split_pts_time - 1.1             #开始截断的时间点 a bit earlier
    split_k_frameinvideo = frame_numbers_np[ind_k_frame]  # 开始截断的 iframe
    vinfo = ffmpegcv.video_info.get_info(video)

    naughty_ffmpeg_copy(
        vinfo, split_pts_time_opt, split_k_frameinvideo, video, video_out_body
    )
    print("done body!")
    fps = vinfo.fps
    vin = ffmpegcv.VideoCapture(video, pix_fmt="nv12")
    vout = ffmpegcv.VideoWriter(
        video_out_neck, codec=vinfo.codec, pix_fmt="nv12", fps=fps
    )
    x265_ = '-x265' if ishevc else '-x264'
    vout.target_pix_fmt = 'yuv420p' + f" {x265_}-params bframes=0 "

    for iframe in range(split_k_frameinvideo):
        ret, frame = vin.read()
        assert ret
        if iframe < num_dehead:
            continue
        vout.write(frame)

    vin.release()
    vout.release()
    time.sleep(2) # wait for ffmpeg to finish
    assert (
        ffmpegcv.video_info.get_info(video_out_neck).count
        == split_k_frameinvideo - num_dehead
    )

    cmd_concat = f'ffmpeg -loglevel warning -i "concat:{video_out_neck}|{video_out_body}" -c copy -r {fps} "{video_concat}" -y '
    os.system(cmd_concat)

    os.remove(video_out_neck)
    os.remove(video_out_body)
    vinfo_dehead = ffmpegcv.video_info.get_info(video_concat)
    assert num_dehead == vinfo.count - vinfo_dehead.count
    print("====OUTPUT====", video_concat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str)
    parser.add_argument("num_dehead", type=int)
    args = parser.parse_args()
    dehead(args.video, args.num_dehead)
