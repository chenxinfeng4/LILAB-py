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
import lilab.cvutils.crop_videofast
import lilab.multiview_scripts.dlcBall_2_landmarks
from subprocess import PIPE, run


def runargs_show(uimessage, args):
    # run the command, and get the stdout and stderr and returncode
    clear(uimessage)
    result = run(args, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    with use_scope(uimessage, clear=True):
        put_text(result.stdout)
        put_error(result.stderr)
        if result.returncode == 0:
            put_success('Success!')
        else:
            put_error('Fail!')


def on_copyconfig(uiinput_path, uimessage):
    clear(uimessage)
    input_path = pin.pin[uiinput_path]
    config_path = osp.join(input_path, 'config_video.py')
    default_config_path = '/home/liying_lab/chenxinfeng/ml-project/multiview_scripts/config_video.py'
    if osp.isdir(input_path):
        # copy without overwrite
        if not osp.isfile(config_path):
            shutil.copyfile(default_config_path, config_path)
            with use_scope(uimessage, clear=True):
                put_success('Copy Success!')
        else:
            with use_scope(uimessage, clear=True):
                put_error('File exists, skip!')
    else:
        with use_scope(uimessage, clear=True):
            put_error('Not a directory!')


def on_crop_video(uiinput_path, uimessage):
    clear(uimessage)
    input_path = pin.pin[uiinput_path]
    
    module = 'lilab.cvutils.crop_video'
    
    args = ['python', '-m', module, input_path]
    runargs_show(uimessage, args)


def app(parent=None):
    if not parent:
        parent = 'crop_video'
        put_scope(parent)
    
    with use_scope(parent):
        c_v_names = names = ['c_v_input_path', 'c_v_message']
        pin.put_input(names[1], label = 'Input path folder')
        put_buttons(['1.copy config', '2. Run'], 
                     onclick=[lambda: on_copyconfig(*c_v_names),
                              lambda:on_crop_video(*c_v_names)])
        put_scope(names[-1])