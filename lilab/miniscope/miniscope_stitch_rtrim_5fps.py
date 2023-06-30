# python -m lilab.miniscope.miniscope_stitich_rtrim_5fps A/B/C/
"""
ffmpeg -i /mnt/ftp.image/ZYQ_2022/data/RAT/LY172/2022_11_06/18_08_49_mdma3/Miniscope/stitch_behavior.avi -c:v copy -f mjpeg - | \
    splitimage | ffmpeg -f image2pipe -r 5 -i - -c:v copy -y /mnt/ftp.image/ZYQ_2022/data/RAT/LY172/2022_11_06/18_08_49_mdma3/Miniscope/stitch_behavior_5fps.avi

ffmpeg -i /mnt/ftp.image/ZYQ_2022/data/RAT/LY172/2022_11_06/18_08_49_mdma3/Miniscope/stitch_behavior.avi -c:v copy -f mjpeg - | \
    splitimage -s 2 | ffmpeg -f image2pipe -r 10 -i - -c:v copy -y /mnt/ftp.image/ZYQ_2022/data/RAT/LY172/2022_11_06/18_08_49_mdma3/Miniscope/stitch_behavior_10fps.avi
"""
# %% imports
import os
import os.path as osp
import ffmpegcv
import re
import argparse
# %%
output_fps = 5
video_path = '/mnt/ftp.image/ZYQ_2022/data/RAT/LY172/2022_11_06/18_08_49_mdma3/Miniscope/stitch_behavior.avi'

def rtrim_5fps_video(video_path):
    # get project path
    video_full_path = osp.abspath(video_path)
    project_basename = osp.basename(osp.dirname(osp.dirname(video_full_path)))

    # %% get video info
    video_info = ffmpegcv.video_info.get_info(video_path)
    assert video_info.fps==20
    assert video_info.fps%output_fps==0
    step = int(video_info.fps/output_fps)
    t_dur = int(video_info.count // video_info.fps);
    print(f"===================== {project_basename} =====================")
    print(f"Rtrim video duration to : {t_dur} s")
    print(f"Convert fps to : {output_fps} fps\n")

    # %% convert video
    pwd_backup = os.getcwd()
    video_dir = osp.dirname(video_path)
    video_basename = osp.basename(video_path)
    video_name, video_ext = osp.splitext(video_basename)
    video_out_basename = f"{video_name}_{output_fps}fps{video_ext}"
    os.chdir(video_dir)

    cmd=(f'ffmpeg -loglevel quiet -t {t_dur} -i {video_basename} -c:v copy -f mjpeg - | '
         f'splitimage -s {step} | '
         f'ffmpeg -loglevel quiet -f image2pipe -r 5 -i - -c:v copy -y {video_out_basename}')
    os.system(cmd)
    os.chdir(pwd_backup)
    return t_dur, osp.join(video_dir, f'{video_out_basename}')


def project_merge(day_dir):
    # list all dirs in day_dir
    dir_list = os.listdir(day_dir)
    dir_list = [d for d in dir_list if osp.isdir(osp.join(day_dir, d))]
    dir_list.sort()
    dir_full_list = [osp.join(day_dir, d) for d in dir_list]

    # check the project dir name is valid as '%H_%M_%S*'
    for d in dir_list:
        assert re.match(r'^\d{2}_\d{2}_\d{2}', d), f"Invalid project dir name: {d}, should be '%H_%M_%S*'"
    
    # check the existance of video file "Miniscope/stitch_behavior.avi" in each project dir
    video_list = [osp.join(d, 'Miniscope', 'stitch_behavior.avi') for d in dir_full_list]
    for video_path in video_list:
        assert osp.exists(video_path), f"Video file not found: {video_path}"
    
    # convert video
    tdurs = []
    video_out_list = []
    for video_path in video_list:
        tdur, video_out = rtrim_5fps_video(video_path)
        tdurs.append(tdur)
        video_out_list.append(video_out)
    
    # merge video
    video_mergeout = osp.join(day_dir, f"stitch_behavior_merge_{output_fps}fps.avi")
    cmdstr = f"""ffmpeg -loglevel quiet -i "concat:{'|'.join(video_out_list)}" -c copy -y {video_mergeout} """
    os.system(cmdstr)
    print(f"Merge video to: {video_mergeout}")

    # merge video info
    video_mergeout_info = osp.join(day_dir, f"stitch_behavior_merge_{output_fps}fps.csv")
    with open(video_mergeout_info, 'w') as f:
        f.write(f"project,tdur(sec)\n")
        for tdur, video_out in zip(tdurs, video_out_list):
            f.write(f"{video_out},{tdur}\n")
    print(f"Merge video info to: {video_mergeout_info}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert video to 5fps')
    parser.add_argument('day_dir', help='day dir')
    args = parser.parse_args()
    project_merge(args.day_dir)
