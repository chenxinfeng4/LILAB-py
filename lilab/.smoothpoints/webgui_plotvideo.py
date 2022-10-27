from pywebio.input import * 
from pywebio.output import * 
from pywebio import pin
from pywebio import start_server
import numpy as np 
import os
import os.path as osp
import glob
import shutil
import cv2
import mmcv
import PIL.Image
import matplotlib
matplotlib.use('Agg')

from lilab.multiview_scripts.ratpoint3d_to_video import main_plot3d
from multiprocessing import Pool
from lilab.cvutils.concat_videopro import concat


scope_msg = 'plotvideo_messages'

video_type_list = ['0_original', '1_impute', '2_outlierfree','3_smooth']
video_file_list = ['rat_points3d_cm.mat', 'rat_points3d_cm_1impute.mat',
                   'rat_points3d_cm_2outlierfree.mat', 'rat_points3d_cm_3smooth.mat']
video_dict = {t: f for t, f in zip(video_type_list, video_file_list)}
def quary_video_from_folder(folder, video_type):
    if folder:
        video_file = osp.join(folder, video_dict[video_type])
    else:
        video_file = None
    return video_file


def on_run():
    with use_scope(scope_msg, clear=True):
        put_text('Loading...')
        folder_white, folder_black = pin.pin.rat_white, pin.pin.rat_black
        assert folder_white or folder_black, 'Please input at least 1 rat folder!'
        if not osp.exists(folder_white) and not osp.exists(folder_black):
            put_error('Both Rat white/black folder Not exist!')
            raise 'Both Rat white/black folder Not exist!'
        #out_file = main_plot3d(folder_white, folder_black)

        select_videos = pin.pin['video_type']
        assert select_videos, 'Please select at least one video type!'
        args = []
        for select_video in select_videos:
            file_white = quary_video_from_folder(folder_white, select_video)
            file_black = quary_video_from_folder(folder_black, select_video)
            assert file_white==None or osp.exists(file_white), f'{file_white} does not exist!'
            assert file_black==None or osp.exists(file_black), f'{file_black} does not exist!'
            args.append((file_white, file_black))
        
        with Pool(len(args)) as pool:
            pool.starmap(main_plot3d, args)

        put_success('Generate success!')


def on_concat():
    with use_scope(scope_msg, clear=True):
        put_text('Loading...')
        folder_white, folder_black = pin.pin.rat_white, pin.pin.rat_black
        assert folder_white or folder_black, 'Please input at least 1 rat folder!'
        if not osp.exists(folder_white) and not osp.exists(folder_black):
            put_error('Both Rat white/black folder Not exist!')
            raise 'Both Rat white/black folder Not exist!'

        select_videos = pin.pin['video_type']
        assert len(select_videos)>1, 'Please select at least two video type!'
        args = []
        for select_video in select_videos:
            file_white = quary_video_from_folder(folder_white, select_video)
            file_black = quary_video_from_folder(folder_black, select_video)
            assert file_white==None or osp.exists(file_white), f'{file_white} does not exist!'
            assert file_black==None or osp.exists(file_black), f'{file_black} does not exist!'
            video = (file_white or file_black).replace('mat', 'mp4')
            args.append(video)
        concat(*args)
        put_success('Concatenate success!')
        

def app(parent=None):
    if not parent:
        parent = 'plot_video'
        put_scope(parent)

    with use_scope(parent):
        pin.put_input('rat_white', label = 'The white rat folder', placeholder='/home/calib/white')
        pin.put_input('rat_black', label = 'The black rat folder', placeholder= '/home/calib/black') 
        pin.put_select('video_type', options= video_type_list,
                        label = 'The video type (multiple choices)',
                        multiple=True)
        put_buttons(['1. Generate each','2. Concatenate'],
                    onclick=[on_run, on_concat])
        
        put_scope(scope_msg)
