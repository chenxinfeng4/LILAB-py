from pywebio.input import * 
from pywebio.output import * 
from pywebio import pin
from pywebio import start_server
import numpy as np 
import glob
import os
import os.path as osp
import glob


scope_msg = 'mmdet_messages'
bathdir = '/home/liying_lab/chenxinfeng/DATA/CBNetV2/'
# pyfile = bathdir+'demo/demo_videos_pkl.py'
pyfile = '-m lilab.mmlab_scripts.mmdet_demo_videos_pkl'
config = bathdir+'mask_rcnn_r101_fpn_2x_coco_bwrat.py'
checkpoint = bathdir+'work_dirs/mask_rcnn_r101_fpn_2x_coco_bwrat/latest.pth'


def on_check():
    with use_scope(scope_msg, clear=True):
        put_text(f'Checking...')
        video_folder = pin.pin['mmdet_video_folder']
        if not osp.exists(video_folder):
            put_error(f'{video_folder} does not exist!')
            return
        assert osp.isfile(config), 'The config file does not exist!'
        assert osp.isfile(checkpoint), 'The checkpoint file does not exist!'
        video_files = glob.glob(osp.join(video_folder, '*.avi'))+glob.glob(osp.join(video_folder, '*.mp4'))
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
    with use_scope(scope_msg, clear=True):
        from lilab.mmlab_scripts.mmdet_demo_videos_pkl import main
        put_text(f'Checking...')
        cudas = pin.pin['choosecuda']
        cudaids = [cudastr.split(':')[-1] for cudastr in cudas]
        assert len(cudaids)>0, 'No cuda device selected!'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cudaids)

        put_text(f'Running...')
        video_folder = pin.pin['mmdet_video_folder']
        main(video_folder, config, checkpoint, len(cudaids))
        put_success('Done!')


def app(parent=None):
    if not parent:
        parent = 'mmdet'
        put_scope(parent)

    with use_scope(parent):
        put_text('Mask R-CNN to seg_pkl')
        pin.put_input('mmdet_video_folder', label='Video folder', placeholder='/home/liying_lab/chenxinfeng/DATA/mmpose/demo/videos/')
        put_buttons(['1. Check', '2. Run'], onclick=[on_check, on_run])

