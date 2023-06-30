# python -m lilab.multiview_scripts.rat2d_kptvideo_script
import os.path as osp
from multiprocessing import Pool
from itertools import cycle
from lilab.multiview_scripts.rat2d_kptvideo import *

work_folder = '/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_30fps/whiteblack/15-37-42-white-copy/dlc2d_seg/'

videos = '15-37-42_output_1_white.avi  15-37-42_output_2_white.avi  15-37-42_output_3_white.avi 15-37-42_output_4_white.avi  15-37-42_output_5_white.avi  15-37-42_output_6_white.avi'.split()
mat_files_white = """15-37-42_output_1_white.mat  15-37-42_output_2_white.mat  15-37-42_output_3_white.mat 
                    15-37-42_output_4_white.mat  15-37-42_output_5_white.mat  15-37-42_output_6_white.mat""".split()
mat_files_black = """15-37-42_output_1_black.mat  15-37-42_output_2_black.mat  15-37-42_output_3_black.mat
                    15-37-42_output_4_black.mat  15-37-42_output_5_black.mat  15-37-42_output_6_black.mat""".split()

fullfile = lambda fs: [osp.join(work_folder, f) for f in fs]
videos = fullfile(videos)
mat_files_white = fullfile(mat_files_white)
mat_files_black = fullfile(mat_files_black)

assert len(videos) == len(mat_files_white) == len(mat_files_black)

with Pool(6) as pool:
    # pool.starmap(main, zip(videos, mat_files_black, mat_files_white))
    # pool.starmap(main, zip(videos, mat_files_black, cycle([None])))
    pool.starmap(main, zip(videos, cycle([None]), mat_files_white))
