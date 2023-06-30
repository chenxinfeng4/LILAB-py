# python -m lilab.cvutils_new.video_put_num  xxx.mp4
import argparse
import ffmpegcv
import tqdm
import os.path as osp
import cv2
resize = (1280, 800)

def convert(video_file, out_file=None):
    if out_file is None:
        out_file = osp.splitext(video_file)[0] + '_num.mp4'
    vid = ffmpegcv.VideoCaptureNV(video_file, gpu=0, resize=resize)
    writer = ffmpegcv.VideoWriterNV(out_file, fps=vid.fps, gpu=0)
    with vid, writer:
        for iframe, frame in enumerate(tqdm.tqdm(vid)):
            cv2.putText(frame, str(iframe), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            writer.write(frame)

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_file')
    parser.add_argument('--out_file', default=None)
    args = parser.parse_args()
    convert(args.video_file, args.out_file)

if __name__ == '__main__':
    __main__()
