# python -m lilab.miniscope.miniscope_clips_stitch_dehead  MINISCOPE_FOLDER/ 
# %%
import argparse
import json
import os
import os.path as osp
import datetime
import glob
#vdir = r'E:\ZYQ_2022\data\RAT\LY150\2022_11_05\14_10_19_baseline'


miniscope_real_fps = 20

def stitch_raw_videos(project_path, fps=miniscope_real_fps):
    Miniscope_dir = osp.join(project_path, 'Miniscope')
    assert osp.exists(Miniscope_dir), 'video not exist'
    pwd_backup = os.getcwd()
    os.chdir(Miniscope_dir)
    vfiles = []
    for i in range(1000):
        vfile = f'{i}.avi'
        if osp.exists(vfile):
            vfiles.append(vfile)
        else:
            break
    else:
        raise 'Too many videos, please check!'

    assert vfiles, 'no video exist'

    cmdstr = 'ffmpeg -loglevel quiet -hide_banner -y -i "concat:{}" -c copy -f mjpeg stitch_raw.mjpeg'.format('|'.join(vfiles))
    os.system(cmdstr)
    cmdstr = 'ffmpeg -loglevel quiet -y -r {} -i stitch_raw.mjpeg -c copy  stitch_raw.avi'.format(fps)
    os.system(cmdstr)
    os.remove('stitch_raw.mjpeg')
    print("[√] Stitch {} videos totally.".format(len(vfiles)))
    os.chdir(pwd_backup)
    return osp.join(Miniscope_dir, 'stitch_raw.avi')


def behavior_register(file_ctime_txt):
    beh_vfiles = []
    beh_datetimes = []
    with open(file_ctime_txt, 'r') as f:
        for line in f:
            assert '\t' in line
            beh_vfile, ctime = line.split('\t')
            beh_datetime = datetime.datetime.strptime(ctime, '%Y-%m-%d %H:%M:%S.%f\n')
            beh_vfiles.append(beh_vfile)
            beh_datetimes.append(beh_datetime)

    assert beh_datetimes, 'No behavior video registed!'
    print("[√] The {} behavior videos registed.".format(len(beh_datetimes)))
    return beh_vfiles, beh_datetimes


def miniscope_behavior_dtime(beh_vfiles, beh_datetimes, project_path, Miniscope_datetime):
    valid_beh_vfiles = []
    valid_beh_datetimes = []
    for beh_vfile, beh_datetime in zip(beh_vfiles, beh_datetimes):
        dt = beh_datetime - Miniscope_datetime
        if dt.days==0 and 0<dt.seconds<30:
            valid_beh_vfiles.append(beh_vfile)
            valid_beh_datetimes.append(beh_datetime)
    assert len(valid_beh_datetimes)>0, 'not found closest behavior video'
    assert len(valid_beh_datetimes)<=1, 'too many matched file'+str(valid_beh_vfiles)
    print("[√] One behavior video matched to miniscope with time range.")

    Behavior_datetime = valid_beh_datetimes[0]
    Behavior_vfile = valid_beh_vfiles[0]
    dt = Behavior_datetime - Miniscope_datetime
    print('[√] Miniscope {} ====({:0.3f}sec)===> Behavior {}'.format(
        osp.basename(project_path), dt.seconds + dt.microseconds/1000000, osp.basename(Behavior_vfile)
    ))
    return dt


def dehead_video(video_in, dt):
    pwd_backup = os.getcwd()
    os.chdir(osp.dirname(video_in))
    cmdstr = 'ffmpeg -loglevel quiet -hide_banner -y -ss {}.{:03} -i "{}" -c copy stitch_behavior.avi'.format(
         dt.seconds, dt.microseconds//1000, osp.basename(video_in))
    os.system(cmdstr)
    print("[√] Dehead miniscope video.")
    os.chdir(pwd_backup)


def read_miniscope_projecttime(project_path):
    metaDatajson = osp.join(project_path, 'metaData.json')
    assert osp.exists(project_path), 'dir not exist'
    assert osp.exists(metaDatajson), 'json not exist'
    with open(metaDatajson, 'rb') as f:
        metaJson = json.load(f)
    jsonST = metaJson['recordingStartTime']
    Miniscope_datetime = datetime.datetime(
        year=jsonST['year'], month=jsonST['month'], day=jsonST['day'], 
        hour=jsonST['hour'], minute=jsonST['minute'], second=jsonST['second'],
        microsecond=int(jsonST['msec']*1000))
    return Miniscope_datetime


def main(project_path, beh_file_ctime_txt):
    beh_vfiles, beh_datetimes = behavior_register(beh_file_ctime_txt)

    if osp.exists(osp.join(project_path, 'notes.csv')):
        project_paths = [project_path]
    else:
        project_paths = [dirpath for dirpath, _, filenames in os.walk(project_path) for filename in filenames if filename=='notes.csv']

    assert project_paths

    for project_path_ in project_paths:
        print('\n--------------{}-------------'.format(project_path_))
        Miniscope_datetime = read_miniscope_projecttime(project_path_)
        dt = miniscope_behavior_dtime(beh_vfiles, beh_datetimes, project_path_, Miniscope_datetime)
        stitch_v = stitch_raw_videos(project_path_)
        dehead_video(stitch_v, dt)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Stitch and dehead Miniscope video clips.')
    parser.add_argument('project_path', type=str, help='Miniscope project folder path')
    parser.add_argument('--beh_file_ctime_txt', type=str, default=r'E:\ZYQ_2022\data\RAT\file_ctime.txt')

    args = parser.parse_args()
    assert os.path.exists(args.project_path), 'file_path not exists'
    assert os.path.exists(args.beh_file_ctime_txt), 'file_path not exists'
    args.project_path = osp.abspath(args.project_path)
    main(args.project_path, args.beh_file_ctime_txt)
