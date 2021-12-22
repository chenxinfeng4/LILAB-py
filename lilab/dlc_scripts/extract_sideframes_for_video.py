# %% import the necessary packages
import os
import glob
import mmcv
import cv2
import os
video_folder = '/home/liying_lab/chenxinfeng/deeplabcut-project/abc-edf-2021-11-23/videos3'
t_frame = 18

# %%
# find the video files
video_files = glob.glob(os.path.join(video_folder, '*.mp4')) + glob.glob(os.path.join(video_folder, '*.avi'))

# sort the video files
video_files.sort()
n_videos = len(video_files)
assert n_videos==6, 'There should be 6 videos in the folder'

# create the output folder inside the video folder, named 'merged_sideframes'
output_folder = os.path.join(video_folder, 'merged_sideframes')
os.makedirs(output_folder, exist_ok=True)

# create the output video file in the output folder
input_v = mmcv.VideoReader(video_files[0])
# get the width and height of the first video
width = input_v.width
height = input_v.height

output_video_file = os.path.join(output_folder, 'merged_sideframes.mp4')
output_v = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

# loop over the video files, and get the frame in t_frame
for i, video_file in enumerate(video_files):
    print('Processing video {}/{}'.format(i+1, n_videos))
    input_v = mmcv.VideoReader(video_file)
    #get the frame rate
    fps = input_v.fps
    frame = input_v[int(t_frame*fps)]
    output_v.write(frame)

output_v.release()

# extract all frames from the output_video 
# and save them in the output_folder
# ffmpeg -i merged_sideframes.mp4 -vf fps=1 merged_sideframes/%d.jpg
os.system('ffmpeg -i "{}" -vf fps=1 "{}/img%03d.png"'.format(output_video_file, output_folder))
