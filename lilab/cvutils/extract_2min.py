import os
import argparse

"""extract the first "DUR" min of a video by ffmpeg"""
def extract_2min(video_path, duration):
    os.system('ffmpeg -i "{}" -t {} -c copy -y "{}"'.format(video_path, duration, video_path[:-4] + "_2min.mp4"))


"""use argparse to get the video path and the minutes to extract"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="path to the video")
    parser.add_argument("--tdur", help="duration of the video to extract", default="00:02:00")
    args = parser.parse_args()
    extract_2min(args.video_path, args.tdur)

"""__name__ == __main__"""
if __name__ == "__main__":
    main()