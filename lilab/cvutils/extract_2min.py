import os
import os.path as osp
import argparse
import glob

"""extract the first "DUR" min of a video by ffmpeg"""
def extract_2min(video_path, duration):
    parent_dir = osp.dirname(osp.abspath(video_path)) + "/headminutes"
    os.makedirs(parent_dir, exist_ok=True)
    video_name = osp.splitext(osp.basename(video_path))[0]
    output_path = osp.join(parent_dir, video_name + "_2min.mp4")
    os.system(f'ffmpeg -i "{video_path}" -t {duration} -c copy -y "{output_path}"')


"""use argparse to get the video path and the minutes to extract"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="path to the video")
    parser.add_argument("--tdur", help="duration of the video to extract", default="00:02:00")
    args = parser.parse_args()
    # check the video_path is a file or a directory
    if os.path.isfile(args.video_path):
        videos = [args.video_path]
    elif os.path.isdir(args.video_path):
        videos = glob.glob(args.video_path + "/*.mp4") +\
                 glob.glob(args.video_path + "/*.avi")
    else:
        raise ValueError("the video path is not a file or a directory")
    # extract the first "DUR" min of each video
    for video in videos:
        extract_2min(video, args.tdur)


"""__name__ == __main__"""
if __name__ == "__main__":
    main()