# python -m lilab.mmlab_scripts.webgui
from cmath import pi
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
import os

from lilab.mmlab_scripts.webgui_mmseg import app as app_mmseg
from lilab.mmlab_scripts.webgui_mmdet import app as app_mmdet
from lilab.mmlab_scripts.webgui_mmpose import app as app_mmpose
from lilab.mmlab_scripts.webgui_cropvideo import app as app_cropvideo


number_gpu = 4
cudalist = ['cuda:'+str(i) for i in range(number_gpu)]
def app():
    
    pin.put_checkbox('choosecuda', 
                    label = 'Choose cuda device',
                    options = cudalist,
                    value = cudalist[:2],
                    inline=True)
    put_tabs([
        {'title': 'crop video', 'content':put_scope('crop_video')},
        {'title': 'mmseg', 'content':put_scope('mmseg')},
        {'title': 'mmdet', 'content': put_scope('mmdet')},
        {'title': 'mmpose', 'content':put_scope('mmpose')},
    ])
    app_mmpose('mmpose')
    app_mmdet('mmdet')
    app_cropvideo('crop_video')
    app_mmseg('mmseg')

if __name__ == '__main__':
    start_server(app, debug=True, port='44321')
