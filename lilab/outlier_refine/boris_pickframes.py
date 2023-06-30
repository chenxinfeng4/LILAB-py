# python -m lilab.outlier_refine.pickframes
# %% imports
import json
import os
import os.path as osp
import re
from lilab.cvutils.extract_frames_by_time import extract_crop
from lilab.cameras_setup import get_view_xywh_1280x800x10 as get_view_xywh
file = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/error.boris'
boris = json.load(open(file, 'r'))
boris_dir = osp.dirname(osp.abspath(file))
# %%
key = 's'
observations = boris['observations']

# %%
def get_abspath_video(v):
    if osp.exists(v):
        return osp.dirname(osp.abspath(v))
    else:
        vfile = osp.join(boris_dir, osp.basename(v))
        if osp.exists(vfile):
            return vfile
        else:
            raise FileNotFoundError(v)


eventdict = {}
for k, value in observations.items():
    video_name = value['file']['1'][0]
    events = value['events']
    events_pick = [event[0] for event in events if event[2] == key]
    video_fullname = get_abspath_video(video_name)
    eventdict[video_fullname] = events_pick


# %%
for v, timestamps in eventdict.items():
    extract_crop(filename=v, 
                    indexby = 'frame',
                    crop_xywh = None,
                    timestamps = timestamps,
                    ipannel = None)


for v, timestamps in eventdict.items():
    v_origin = re.compile(r"_\d+.mp4").sub('.mp4', v)
    ipannel = int(re.findall(r"_(\d+).mp4", v)[0])
    extract_crop(   filename=v_origin, 
                    indexby = 'frame',
                    crop_xywh = get_view_xywh()[ipannel],
                    timestamps = timestamps,
                    ipannel = ipannel)