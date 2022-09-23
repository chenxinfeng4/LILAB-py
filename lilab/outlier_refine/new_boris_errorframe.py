# python -m lilab.outlier_refine.new_boris_errorframe A/B/C
# python -m lilab.outlier_refine.new_boris_errorframe D.mp4 E.mp4 F.mp4
# %%
import json
import argparse
import glob
import os.path as osp
import pprint
import ffmpegcv.video_info

# %%
boris_project = {
    "time_format":"hh:mm:ss",
    "project_date":"2022-05-09T10:50:47",
    "project_name":"errorframe",
    "project_description":"",
    "project_format_version":"7.0",
    "subjects_conf":{},
    "behaviors_conf":{
        "0":{
            "type":"Point event",
            "key":"s",
            "code":"s",
            "description":"",
            "category":"",
            "modifiers":"",
            "excluded":"",
            "coding map":""
        }
    },
    "observations":{},
    "behavioral_categories":[],
    "independent_variables":{},
    "coding_map":{},
    "behaviors_coding_map":[],
    "converters":{}
}

def get_observation(video):
    basename = osp.basename(video)
    nakename = osp.splitext(osp.basename(video))[0]
    vinfo = ffmpegcv.video_info.get_info(video)
    media_info = {
        "length":   {nakename: vinfo.duration},
        "fps":      {nakename: vinfo.fps},
        "hasVideo": {nakename: True},
        "hasAudio": {nakename: False},
        "offset"  : {"1": 0}
    }
    observation = {
        nakename:{
        "file":{
            "1":[basename],
            "2":[],
            "3":[],
            "4":[],
            "5":[],
            "6":[],
            "7":[],
            "8":[]
        },
        "type":"MEDIA",
        "date":"2022-05-09T10:51:50",
        "description":"",
        "time offset":0.0,
        "events":[],
        "observation time interval":[0,0],
        "independent_variables":{},
        "visualize_spectrogram":False,
        "visualize_waveform":False,
        "close_behaviors_between_videos":False,
        "media_info":media_info
        }
    }
    return observation


def get_observations_collect_project(videos):
    if isinstance(videos, str):
        videos = [videos]
    observations = {}
    for video in videos:
        observation = get_observation(video)
        observations.update(observation)
    project_now = boris_project.copy()
    project_now['observations'] = observations
    basedir = osp.dirname(osp.abspath(videos[0]))
    project_file = osp.join(basedir, 'errorframe.boris')
    with open(project_file, 'w') as f:
        json.dump(project_now, f, indent=2)
    return project_file


def search_videos_in_dir(dir):
    videos = [file for file in glob.glob(osp.join(dir, '*.mp4'))
                if file[-5].isdigit()]
    return videos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('videos_or_dir', nargs='+', type=str)
    args = parser.parse_args()
    videos = []
    if len(args.videos_or_dir) == 1 and osp.isdir(args.videos_or_dir[0]):
        videos = search_videos_in_dir(args.videos_or_dir[0])
    elif osp.isfile(args.videos_or_dir[0]):
        videos = args.videos_or_dir
    else:
        raise ValueError('videos_or_dir must be a directory or a video file')
    pprint.pprint([osp.basename(video) for video in videos])
    project_file = get_observations_collect_project(videos)
    print("=======project file:", project_file)