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
from lilab.smoothpoints.LSTM_point3d_impute_train import main

scope_msg = 'lstm_train_messages'

def on_run():
    input_paths = pin.pin['cc_input_textarea'].split()
    for input_path in input_paths:
        if not osp.exists(input_path):
            with use_scope(scope_msg, clear=True):
                put_error(f'{input_path} does not exist!')
            return
    with use_scope(scope_msg, clear=True):
        put_text(f'Training on {len(input_paths)} files...')
        msgstring = main(input_paths)
        put_text(msgstring)


def app(parent=None):
    if not parent:
        parent = 'lstm_train'
        put_scope(parent)

    with use_scope(parent):
        pin.put_textarea('cc_input_textarea', 
                         label = 'The training "points3d.mat" files',
                         rows=6, code={'lineNumbers' : True})
        put_button('Run', onclick=on_run)
        put_scope(scope_msg)
