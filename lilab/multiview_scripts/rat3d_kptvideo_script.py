# python -m lilab.multiview_scripts.rat3d_kptvideo_script
import os.path as osp
from multiprocessing import Pool
from lilab.multiview_scripts.rat3d_kptvideo import pose_json, main

work_folder = '/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_30fps/whiteblack/15-37-42-white-copy/dlc3d/'

videos = '15-37-42_output_1.mp4  15-37-42_output_2.mp4  15-37-42_output_3.mp4 15-37-42_output_4.mp4  15-37-42_output_5.mp4  15-37-42_output_6.mp4'.split()
mat_files_white = ["rat_points3d_cm-white.mat"] * len(videos)
mat_files_black = ["rat_points3d_cm-black.mat"] * len(videos)
pose = [pose_json] * len(videos)

fullfile = lambda fs: [osp.join(work_folder, f) for f in fs]
videos = fullfile(videos)
mat_files_white = fullfile(mat_files_white)
mat_files_black = fullfile(mat_files_black)


assert len(videos) == len(mat_files_white) == len(mat_files_black)

with Pool(6) as pool:
    pool.starmap(main, zip(videos, pose, mat_files_black, mat_files_white))
