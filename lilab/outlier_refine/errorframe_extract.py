# python -m lilab.outlier_refine.errorframe_extract
# %%
import json
import mmcv
import cv2
import os.path as osp
import os
from lilab.cvutils.auto_find_subfolderimages import find_subfolderfiles
import importlib
import ffmpegcv
import ffmpegcv.ffmpeg_framepick
from lilab.cameras_setup import get_view_xywh_1280x800x10 as get_view_xywh

# %%
jsonfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/out.json'
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
    picker = ffmpegcv.ffmpeg_framepick.FFmpegFramePick.VideoFramePick(full_vfile, None, 'bgr24', crop_xywh, None, None, None)
    picker.codec = picker.codec+'_cuvid'
    print(full_vfile)
    framestamps.sort()
    outframes = picker[framestamps]
    for iframe in framestamps:
        img = outframes[iframe]
        if img is None: continue
        mmcv.imwrite(img, outprefix + '_{0:05}.png'.format(iframe))


for vfile, full_vfile, outprefix in zip(vfiles, full_vfiles, vfile_outprefixs):
    framestamps = data[vfile]
    extract_frames(full_vfile, framestamps, outprefix)

# %%
importlib.reload(ffmpegcv.ffmpeg_framepick)

picker = ffmpegcv.ffmpeg_framepick.FFmpegFramePick.VideoFramePick(full_vfile, None, 'bgr24', None, None, None, None)
outframes = picker[framestamps]

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
