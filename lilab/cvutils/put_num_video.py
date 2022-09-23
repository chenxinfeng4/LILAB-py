# python -m lilab.cvutils.put_num_video A.mp4
# %% imports
import cv2
import ffmpegcv
import argparse
import tqdm
import os.path as osp

def convert(video_in):
    vid = ffmpegcv.VideoCaptureNV(video_in)
    video_out = video_in.replace('.mp4', f'_num.mp4')
    vidout = ffmpegcv.VideoWriterNV(video_out, 
                                    gpu = 0,
                                    codec='h264', 
                                    fps=vid.fps)
    for i, frame in enumerate(tqdm.tqdm(vid)):
        frame = cv2.putText(frame, str(i), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        vidout.write(frame)

    vid.release()
    vidout.release()

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, default=None, help='path to video')
    args = parser.parse_args()
    video_path = args.video_path
    assert osp.exists(video_path), 'video_path not exists'
    convert(video_path)
