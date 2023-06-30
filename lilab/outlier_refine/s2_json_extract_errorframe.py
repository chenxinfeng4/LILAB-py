# python -m lilab.outlier_refine.s2_json_extract_errorframe
# %%
import json
import mmcv
import cv2
import os.path as osp
import os
from lilab.cvutils.auto_find_subfolderimages import find_subfolderfiles
import tqdm
import ffmpegcv
import ffmpegcv.ffmpeg_framepick
from lilab.cameras_setup import get_view_xywh_1280x800x10 as get_view_xywh

# %%
jsonfile = 'frames_seperate/out.json'
with open(jsonfile, 'r') as f:
    data = json.load(f)

# %%
vfiles = list(data.keys())
full_vfiles = find_subfolderfiles(jsonfile, vfiles)

outfolder = osp.join(osp.dirname(jsonfile), 'errorframe')
vfile_outprefixs = [osp.join(outfolder,osp.basename(osp.splitext(v)[0])+'') for v in vfiles]
os.makedirs(outfolder, exist_ok=True)

# %%
def extract_frames(full_vfile, framestamps, outprefix, crop_xywh=None):
    cap = ffmpegcv.VideoCaptureNV(full_vfile, crop_xywh=crop_xywh)
    framestamps.sort()
    framestamps_set = set(framestamps)
    for iframe, img in enumerate(cap):
        if iframe in framestamps_set:
            mmcv.imwrite(img, outprefix + '_{0:05}.jpg'.format(iframe))
        elif iframe > framestamps[-1]:
            break
        else:
            continue

    cap.release()

for vfile, full_vfile, outprefix in zip(tqdm.tqdm(vfiles), full_vfiles, vfile_outprefixs):
    framestamps = data[vfile]
    extract_frames(full_vfile, framestamps, outprefix)


# %%
views = get_view_xywh()
vfiles = list(data.keys())
vfile_canvas = [v[:-6]+'.mp4' for v in vfiles]
iviews_canvas = [int(v[-5]) for v in vfiles]
full_vfile_canvas = find_subfolderfiles(jsonfile, vfile_canvas)

outfolder = osp.join(osp.dirname(jsonfile), 'errorframe_canvas')
vfile_outprefixs = [osp.join(outfolder,osp.basename(osp.splitext(v)[0])+'') for v in vfiles]
os.makedirs(outfolder, exist_ok=True)
for vfile, full_vfile, outprefix, iview in zip(vfiles, full_vfile_canvas, vfile_outprefixs, iviews_canvas):
    crop_xywh = views[iview]
    framestamps = data[vfile]
    extract_frames(full_vfile, framestamps, outprefix, crop_xywh)
