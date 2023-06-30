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

from lilab.multiview_scripts.ratpoint3d_to_video import main_plot3d
# /home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_side6/rat/rat_points3d_cm_impute.mat
# /home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_side6/ratBlack/rat_points3d_cm_hybrid.mat

scope_msg = 'tworat_messages'

def on_plot_rats():
    with use_scope('scope_msg', clear=True):
        matlab_white, matlab_black = pin.pin.rat_white, pin.pin.rat_black
        if not osp.exists(matlab_white) and not osp.exists(matlab_black):
            put_error('Both Rat white/black file Not exist!')
            raise 'Both Rat white/black file Not exist!'
        out_file = main_plot3d(matlab_white, matlab_black)

        if osp.exists(out_file):
            put_success('Success!')
            put_text('[The output file is >>] {}'.format(out_file))
            vid = mmcv.VideoReader(out_file)
            nframe = len(vid)
            nchoose = min(nframe, 4)
            ichoose = np.sort(np.random.choice(nframe, nchoose, replace=False))
            for i in ichoose:
                img = vid[i]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                im_pil = PIL.Image.fromarray(img)
                put_image(im_pil)
        else:
            put_error('Fail!')


def app(parent=None):
    if not parent:
        parent = 'outrat'
        put_scope(parent)

    with use_scope(parent):
        pin.put_input('rat_white', label = 'The white rat', placeholder='/home/calib/rat_white_point3d_cm.mat')
        pin.put_input('rat_black', label = 'The rat video folder', placeholder= '/home/calib/rat_black_point3d_cm.mat') 
        put_button('Plot rats',  onclick=on_plot_rats)
        put_scope(scope_msg)

