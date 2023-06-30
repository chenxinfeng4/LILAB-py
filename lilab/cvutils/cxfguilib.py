#!/usr/bin/python
import os, glob, sys
import os.path
import importlib.util

#http://timgolden.me.uk/python/win32_how_do_i/browse-for-a-folder.html
def uigetfolder():
    path = input("select a folder >> ")
    return path
        

def ui_assert(eqtion, msg):
    if not eqtion:
        input("Error错误 ")
        sys.exit(1)

def getffmpeg():
    ffmpeg_path_d = r"D:\L_ffmpeg\ffmpeg\bin\ffmpeg.exe"
    ffmpeg_path_linux = '/usr/bin/ffmpeg'
    ffmpeg_path_loc = os.path.join(os.path.split(sys.argv[0])[0], "ffmpeg.exe")
    if os.path.isfile(ffmpeg_path_loc):
        ffmpeg_path = ffmpeg_path_loc
    elif os.path.isfile(ffmpeg_path_linux):
        ffmpeg_path = ffmpeg_path_linux
    elif os.path.isfile(ffmpeg_path_d):
        ffmpeg_path = ffmpeg_path_d
    else:
        ui_assert(False, "找不到FFMPEG.EXE, 请拷贝到程序文件夹!")
    return ffmpeg_path
        
def getfoldconfigpy(folder):
    configfile = os.path.join(folder, "config_video.py")
    assert os.path.isfile(configfile), 'No [{}] such config file'.format(configfile)
    spec=importlib.util.spec_from_file_location("config_video",configfile)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo
    