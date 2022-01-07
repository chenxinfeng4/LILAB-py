# concat two video file by ffmpeg
import os
import os.path as osp
import argparse

def concat(videopath1, videopath2):
    # concat the videos horizontally
    output_path = osp.join(osp.dirname(videopath1), "concat.mp4")
    os.system(f'ffmpeg -i "{videopath1}" -i "{videopath2}" -filter_complex "[0:v][1:v]hstack=inputs=2[out]" -map "[out]" -c:v libx264  "{output_path}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("videopath1", help="path to the first video")
    parser.add_argument("videopath2", help="path to the second video")
    args = parser.parse_args()
    concat(args.videopath1, args.videopath2)