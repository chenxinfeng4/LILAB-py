from pywebio.input import * 
from pywebio.output import * 
from pywebio import pin
from pywebio import start_server
import numpy as np 
import glob
import os
import os.path as osp
import glob
import shutil
import cv2
import mmcv
import PIL.Image


scope_msg = 'mmlab_messages'
bathdir = '/home/liying_lab/chenxinfeng/DATA/mmpose/'
pyfile = bathdir+'demo/demo_videos_pkl_pro.py'
config = bathdir+'hrnet_w32_coco_udp_rat.py'
checkpoint = bathdir+'work_dirs/hrnet_w32_coco_udp_rat/latest.pth'
dlc_config = '/home/liying_lab/chenxinfeng/deeplabcut-project/abc-edf-2021-11-23/config.yaml'

def on_check():
    with use_scope(scope_msg, clear=True):
        put_text(f'Checking...')
        video_folder = pin.pin['video_folder']
        if not osp.exists(video_folder):
            put_error(f'{video_folder} does not exist!')
            return
        assert osp.isfile(pyfile), 'The demo file does not exist!'
        assert osp.isfile(config), 'The config file does not exist!'
        assert osp.isfile(checkpoint), 'The checkpoint file does not exist!'
        video_files = glob.glob(osp.join(video_folder, '*.avi'))
        video_files_nake = [osp.basename(v) for v in video_files]
        if len(video_files_nake) == 0:
            put_error(f'{video_folder} does not contain any video files!')
            return
        else:
            put_text('\n'.join(video_files_nake))

        cmd = 'choosecuda 0\n'+ \
              f'python {pyfile} {video_folder} {config} {checkpoint} 4' + \
              '\n'
        put_code(cmd, rows=4)
        put_success('Done!')
        

def on_run():
    from lilab.mmlab_scripts.mmpose_demo_videos_pkl import main as main_to_pkl
    with use_scope(scope_msg, clear=True):
        put_text(f'Checking...')
        cudas = pin.pin['choosecuda']
        cudaids = [cudastr.split(':')[-1] for cudastr in cudas]
        assert len(cudaids)>0, 'No cuda device selected!'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cudaids)

        put_text(f'Running...')
        video_folder = pin.pin['video_folder']
        main_to_pkl(video_folder, config, checkpoint, len(cudaids))
        put_success('Done!')
        

def on_to_csv():
    from lilab.mmlab_scripts.mmpose_to_deeplabcut import main as main_to_deeplabcut
    with use_scope(scope_msg, clear=True):
        put_text(f'To csv...')
        video_folder = pin.pin['video_folder']
        pkl_files = glob.glob(osp.join(video_folder, '*.pkl'))
        nfile = len(pkl_files)
        for i, pkl_file in enumerate(pkl_files):
            pkl_file_nake = osp.basename(pkl_file)
            main_to_deeplabcut(dlc_config, pkl_file)
            put_text(f'{i+1}/{nfile}: {pkl_file_nake})')
            
        put_success('Done!')


def on_clean():
    with use_scope(scope_msg, clear=True):
        put_text(f'Cleaning...')
        video_folder = pin.pin['video_folder']
        meta_files = glob.glob(osp.join(video_folder, '*.pickle'))
        h5_files = glob.glob(osp.join(video_folder, '*.h5'))
        for meta_file in meta_files:
            os.remove(meta_file)
        for h5_file in h5_files:
            os.remove(h5_file)
        put_success('Done!')


def app(parent=None):
    if not parent:
        parent = 'mmlab'
        put_scope(parent)

    with use_scope(parent):
        pin.put_input('video_folder', label='Video folder', placeholder='/home/liying_lab/chenxinfeng/DATA/mmpose/demo/videos/')
        put_buttons(['1. Check', '2. Run', '3. To CSV', '3_5. Clean meta&h5'], 
                    onclick=[on_check, on_run, on_to_csv, on_clean])
