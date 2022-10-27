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
from lilab.smoothpoints.outlierfree_smooth_point3d import main


scope_msg = 'outlier_smooth_messages'

def on_run():
    input_paths = pin.pin['input_textarea_outlier_smooth'].split()
    for input_path in input_paths:
        if not osp.exists(input_path):
            with use_scope(scope_msg, clear=True):
                put_error(f'{input_path} does not exist!')
            return
        if not input_path.endswith('points3d_cm_1impute.mat'):
            with use_scope(scope_msg, clear=True):
                put_error(f'{input_path} is not a "points3d_1impute.mat" file!')
            return

    with use_scope(scope_msg, clear=True):
        put_text(f'Training on {len(input_paths)} files...')
        for input_path in input_paths:
            put_text(input_path[-20:])
            main(input_path)
        put_success('Done!')


def app(parent=None):
    if not parent:
        parent = 'outlier_smooth'
        put_scope(parent)

    with use_scope(parent):
        pin.put_textarea('input_textarea_outlier_smooth', 
                         label = 'The training "points3d_1impute.mat" files',
                         rows=6, code={'lineNumbers' : True})
        put_button('Run', onclick=on_run)
        put_scope(scope_msg)

