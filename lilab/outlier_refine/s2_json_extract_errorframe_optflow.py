# python -m lilab.outlier_refine.errorframe_extract
# %%
import json
import mmcv
import os.path as osp
import os
import tqdm
from lilab.cvutils.auto_find_subfolderimages import find_subfolderfiles
import numpy as np
import ffmpegcv
import ffmpegcv.ffmpeg_framepick
from lilab.cameras_setup import get_view_xywh_1280x800x10 as get_view_xywh

frame_win = [3, 3]
# %%
jsonfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/out.json'
with open(jsonfile, 'r') as f:
    data = json.load(f)

# %%
vfiles = list(data.keys())
full_vfiles = find_subfolderfiles(jsonfile, vfiles)

outfolder = osp.join(osp.dirname(jsonfile), 'errorframe_optflow')
vfile_outprefixs = [osp.join(outfolder,osp.basename(osp.splitext(v)[0])+'') for v in vfiles]
os.makedirs(outfolder, exist_ok=True)

# %%
def extract_frames(full_vfile, framestamps, outprefix, crop_xywh=None):
    cap = ffmpegcv.VideoCaptureNV(full_vfile, crop_xywh=crop_xywh)
    framestamps.sort()
    framestamps = np.array(framestamps)
    framestamps_pre = framestamps - frame_win[0]
    framestamps_post = framestamps + frame_win[1]
    container_pre = []
    container_post = []
    container_now = []
    for iframe, img in enumerate(cap):
        img = img[:,:,0]
        if iframe in framestamps_pre:
            container_pre.append(img)
        if iframe in framestamps_post:
            container_post.append(img)
        if iframe in framestamps:
            container_now.append(img)
        if iframe > framestamps_post[-1]:
            break
        else:
            continue
    
    if len(container_post) < len(container_now):
        container_post.append(container_now[-1])
    if len(container_pre) < len(container_now):
        container_pre.insert(0, container_now[0])

    cap.release()

    for i, iframe in enumerate(framestamps):
        img = np.stack((container_pre[i], container_now[i], container_post[i]), axis=2)
        mmcv.imwrite(img, outprefix + '_{0:05}.jpg'.format(iframe))


# for vfile, full_vfile, outprefix in zip(tqdm.tqdm(vfiles), full_vfiles, vfile_outprefixs):
#     framestamps = data[vfile]
#     extract_frames(full_vfile, framestamps, outprefix)


# %%
views = get_view_xywh()
vfiles = list(data.keys())
vfile_canvas = [v[:-6]+'.mp4' for v in vfiles]
iviews_canvas = [int(v[-5]) for v in vfiles]
full_vfile_canvas = find_subfolderfiles(jsonfile, vfile_canvas)

outfolder = osp.join(osp.dirname(jsonfile), 'errorframe_canvas_optflow')
vfile_outprefixs = [osp.join(outfolder,osp.basename(osp.splitext(v)[0])+'') for v in vfiles]
os.makedirs(outfolder, exist_ok=True)
for vfile, full_vfile, outprefix, iview in zip(vfiles, full_vfile_canvas, vfile_outprefixs, iviews_canvas):
    crop_xywh = views[iview]
    framestamps = data[vfile]
    extract_frames(full_vfile, framestamps, outprefix, crop_xywh)
