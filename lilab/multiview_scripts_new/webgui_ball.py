from pywebio.input import * 
from pywebio.output import * 
from pywebio import pin
import numpy as np
import pickle
import os.path as osp

from lilab.multiview_scripts_new.s2_matpkl2ballpkl import convert as convert_matpkl2ballpkl
from lilab.multiview_scripts_new.s3_ballpkl2calibpkl import main_calibrate
from lilab.multiview_scripts_new.s4_matpkl2matcalibpkl import convert as convert_matpkl2matcalibpkl
from lilab.multiview_scripts_new.s5_show_calibpkl2video import main_showvideo
import pickle

"""
//mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/OXTRHETxKO/ball/2022-05-10_05-11ball.mp4
/home/liying_lab/chenxinfeng/DATA/multiview-project/tph2ko/ball.mp4
00:00:01
00:00:11
00:00:18
00:00:29
00:00:39
"""
scope_msg = 'ballcalib_messages'
scope_all_in_one_msg = 'all_in_one_messages'

def get_seconds_from_str(str):
    if ':' in str:
        h, m, s = str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    else:
        return float(str)


def on_a1_predict_ball():
    with use_scope(scope_msg, clear=True):
        python = '/home/liying_lab/chenxinfeng/.conda/envs/mmpose/bin/python'
        cmd =  f'{python} -m lilab.multiview_scripts_new.s1_ballvideo2matpkl "{pin.pin.ballvideo}"'
        put_success(cmd)


def on_a2_matpkl2ballpkl():
    matpkl = osp.splitext(pin.pin.ballvideo)[0] + '.matpkl'
    with use_scope(scope_msg, clear=True):
        put_text('Reading global times...\n')
        names = ['t_global_ball'+str(i) for i in range(1, 6)]   
        checkinputs = [pin.pin[name] for name in names]
        if not all(checkinputs):
            put_error('Please fill in all the time points!')
            raise Exception('Please fill in all the time points!')
        put_text('Converting ...\n')
        
        global_times = [get_seconds_from_str(pin.pin[name]) for name in names]
        if global_times != sorted(global_times):
            put_error('Please fill in the time points in ascending order!')
            raise Exception('Please fill in the time points in ascending order!')

        matpkl = osp.splitext(pin.pin.ballvideo)[0] + '.matpkl'
        outfile = convert_matpkl2ballpkl(matpkl, global_times)
        data = pickle.load(open(outfile, 'rb'))
        put_text(outfile)
        put_text(list(data.keys()))
        put_success('Done')


def on_a3_ballpkl2calibpkl():
    with use_scope(scope_msg, clear=True):
        put_text('Converting to matpkl...\n')
        ballpkl = osp.splitext(pin.pin.ballvideo)[0] + '.ballpkl'
        outfile = main_calibrate(ballpkl)
        put_text(outfile)
        put_success('Done!')


def on_a4_show_calibpkl2video():
    matpkl = osp.splitext(pin.pin.ballvideo)[0] + '.matpkl'
    calibpkl = osp.splitext(pin.pin.ballvideo)[0] + '.calibpkl'
    print("matpkl: ", matpkl)
    print("calibpkl: ", calibpkl)
    with use_scope(scope_msg, clear=True):
        put_text('Converting ...\n')
        outfile = convert_matpkl2matcalibpkl(matpkl, calibpkl)
        put_text(outfile)
        put_success('Done!')
        put_text('Showing video...\n')
        main_showvideo(outfile)
        put_success('Done!')


def on_a5_all_in_one():
    with use_scope(scope_all_in_one_msg, clear=True):
        put_text('Wait for a while (~4 min)...\n')
        put_text('1_predict ball...\n')
        on_a1_predict_ball()
        put_text('2_matpkl to ballpkl...\n')
        on_a2_matpkl2ballpkl()
        put_text('3_ballpkl to calibpkl...\n')
        on_a3_ballpkl2calibpkl()
        put_text('4_show calibpkl to video...\n')
        on_a4_show_calibpkl2video()
        put_success('Done!')


def app(parent=None):
    if not parent:
        parent = 'ballglobal'
        put_scope(parent)
    
    with use_scope(parent):
        pin.put_input('ballvideo', label='The video of ball')
        put_table([
            ['Location', '1', '2', '3', '4', '5'],
            [ 'Time',
              pin.put_input('t_global_ball1', placeholder='00:00:01'),
              pin.put_input('t_global_ball2', placeholder='00:00:02'),
              pin.put_input('t_global_ball3', placeholder='00:00:03'),
              pin.put_input('t_global_ball4', placeholder='00:00:04'),
              pin.put_input('t_global_ball5', placeholder='00:00:05')]
        ])
        put_buttons(['1. predict ball to matpkl',
                     '2. matpkl to ballpkl (fetch global time)',
                     '3. ballpkl to calibpkl (calibration)',
                     '4. show ball in video (out video)'],
                    onclick=[
                        on_a1_predict_ball,
                        on_a2_matpkl2ballpkl,
                        on_a3_ballpkl2calibpkl,
                        on_a4_show_calibpkl2video,
                     ])
        put_button('2-4. All in one (calib and video)', onclick=on_a5_all_in_one)
        put_scope(scope_all_in_one_msg)
        put_scope(scope_msg)

