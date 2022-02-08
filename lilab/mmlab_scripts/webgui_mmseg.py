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


scope_msg = 'mmseg_messages'
bathdir = '/home/liying_lab/chenxinfeng/DATA/mmsegmentation/'
pyfile = bathdir+'demo/demo_videos_pkl.py'
config = bathdir+'upernet_swin_small_patch4_window7_bwrat.py'
checkpoint = bathdir+'work_dirs/upernet_swin_small_patch4_window7_bwrat/latest.pth'


def on_check():
    with use_scope(scope_msg, clear=True):
        put_text(f'Checking...')
        video_folder = pin.pin['mmseg_video_folder']
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
              f'python {pyfile} {video_folder} {config} {checkpoint}' + \
              '\n'
        put_code(cmd, rows=4)
        put_success('Done!')
        

def on_run():
    from lilab.mmlab_scripts.mmseg_demo_videos_pkl import main
    with use_scope(scope_msg, clear=True):
        put_text(f'Checking...')
        cudas = pin.pin['choosecuda']
        cudaids = [cudastr.split(':')[-1] for cudastr in cudas]
        assert len(cudaids)>0, 'No cuda device selected!'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cudaids)

        put_text(f'Running...')
        video_folder = pin.pin['video_folder']
        main(video_folder, config, checkpoint, len(cudaids))
        put_success('Done!')


def app(parent=None):
    if not parent:
        parent = 'mmlab'
        put_scope(parent)

    with use_scope(parent):
        pin.put_input('mmseg_video_folder', label='Video folder', placeholder='/home/liying_lab/chenxinfeng/DATA/mmpose/demo/videos/')
        put_buttons(['1. Check', '2. Run'], onclick=[on_check, on_run])

