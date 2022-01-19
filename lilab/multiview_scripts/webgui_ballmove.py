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

"""
/home/liying_lab/chenxinfeng/deeplabcut-project/balllabel/
/home/liying_lab/chenxinfeng/DATA/multiview-project/tph2ko/ball.mp4
00:00:40
"""
subdirname = 'ballmove'
scope_msg = 'ballmove_messages'
multiview_template_path = '/home/liying_lab/chenxinfeng/ml-project/multiview_scripts/multiview_template'

def get_seconds_from_str(str):
    if ':' in str:
        h, m, s = str.split(':')
        return int(h) * 3600 + int(m) * 60 + int(s)
    else:
        return int(str)

def on_crop_video():
    with use_scope('ballglobal_messages', clear=True):        
        ballvideopath = pin.pin.move_ball_path
        balldir = osp.dirname(ballvideopath)
        if not osp.exists(ballvideopath):
            put_error('The global ball video does not exist!')
            raise Exception('The global ball video does not exist!')
            return
        
        # use ffmpeg
        # ffmpeg -i <input> -filter:v "select='eq(t,3)-eq(t,5)-eq(t,14)'" <output>
        selecttimestr = pin.pin['start_time'] if pin.pin['start_time'] else '00:00:01'

        outdir = osp.join(balldir, subdirname)
        os.makedirs(outdir, exist_ok=True)
        shutil.copy(balldir + '/config_video.py', outdir)
        outballvideopath = osp.join(outdir, 'ball_move.mp4')
        os.system(f'ffmpeg -i "{ballvideopath}" -ss {selecttimestr} -c:v copy -y "{outballvideopath}"')
        put_text('Extracting images done! Crop the images...')
        lilab.cvutils.crop_videofast.main(outdir)

        put_success('Done!')


def on_deeplabcut_predict_ball():
    import deeplabcut as dlc
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    with use_scope('ballglobal_messages', clear=True):
        videos_folder=osp.dirname(pin.pin.move_ball_path) + '/' + subdirname
        config_name = pin.pin.dlc_project_moveball
        if osp.isdir(config_name):
            config_name = osp.join(config_name, 'config.yaml')
        if not osp.isfile(config_name):
            put_error('The Deeplabcut project does not exist!')
            raise Exception('The Deeplabcut project does not exist!')
            return
        
        put_text('Processing:',videos_folder)
        videos = glob.glob(videos_folder+'/*output*.mp4')
        videos_labeled = glob.glob(videos_folder+'/*labeled.mp4')
        videos_orginal = list(set(videos) - set(videos_labeled))
        assert len(videos_orginal) == 6
        dlc.analyze_videos(config_name, videos=videos_orginal, save_as_csv=True)


def on_csv_to_landmarkjson():
    with use_scope('ballglobal_messages', clear=True):
        csv_folder = osp.dirname(pin.pin.move_ball_path) + '/' + subdirname
        landmarkjson_path = osp.join(csv_folder, 'landmarks.json')
        put_text('Converting csv to landmarkjson...')
        lilab.multiview_scripts.dlcBall_2_landmarks.main(csv_folder, landmarkjson_path)
        put_text("""python -m lilab.multiview_scripts.dlcBall_2_landmarks \
            $BALL_MOVE \
            --out $PROJECT_3D/landmarks.json
        """)
        put_success('Done!')


def on_copy_configs():
    os.chdir(osp.dirname(pin.pin.move_ball_path) + '/' + subdirname)
    def copy_without_overwrite(srcs):
        for src in srcs:
            if not osp.isfile(osp.basename(src)):
                shutil.copy(src, '.')

    globalfiles = glob.glob('../ballglobal/*view*.png') + \
                    ['../ballglobal/landmarks_global_pixel.json']
    copy_without_overwrite(globalfiles)

    templates = ['setup.json', 'intrinsics_rational.json', 'landmarks_global_world.json',
                 'filenames.json', 'ba_config_rational.json']
    templates = [osp.join(multiview_template_path, f) for f in templates]
    copy_without_overwrite(templates)

    with use_scope('ballglobal_messages', clear=True):
        put_text('Copying configs done!')

def chdir_to_ballmove():
    os.chdir(osp.dirname(pin.pin.move_ball_path) + '/' + subdirname)

def on_compute_relative_poses():
    chdir_to_ballmove()
    import multiview_calib.scripts.compute_relative_poses_robust as mc
    mc.main(setup = 'setup.json',
            intrinsics = 'intrinsics_rational.json',
            landmarks = 'landmarks.json',
            filenames = 'filenames.json',
            dump_images= True,
            method = 'lmeds',
            max_paths = 5)
    with use_scope('ballglobal_messages', clear=True):
        put_text("""python  -m multiview_calib.scripts.compute_relative_poses_robust \
	-s setup.json \
	-i intrinsics_rational.json \
	-l landmarks.json \
	-m lmeds -n 5 \
	-f filenames.json \
	--dump_images 
        """)
        put_success('Done!')   

def on_concatenate_relative_poses():
    chdir_to_ballmove()
    import multiview_calib.scripts.concatenate_relative_poses as mc
    mc.main(setup = 'setup.json',
            relative_poses = 'output/relative_poses/relative_poses.json',
            dump_images= True)
    with use_scope('ballglobal_messages', clear=True):
        put_text("""python  -m multiview_calib.scripts.concatenate_relative_poses \
	-s setup.json \
	-r output/relative_poses/relative_poses.json \
	--dump_images 
        """)
        put_success('Done!')    

def on_bundle_adjustment():
    chdir_to_ballmove()
    import multiview_calib.scripts.bundle_adjustment as mc
    mc.main(setup = 'setup.json',
            intrinsics='intrinsics_rational.json',
            extrinsics='output/relative_poses/poses.json',
            landmarks='landmarks.json',
            filenames='filenames.json',
            dump_images=True,
            config = 'ba_config_rational.json')
    with use_scope('ballglobal_messages', clear=True):
        put_text("""python  -m multiview_calib.scripts.bundle_adjustment \
	-s stup.json \
	-i intrinsics_rational.json \
	-e output/relative_poses/poses.json \
	-l landmarks.json  \
	-f filenames.json \
	--dump_images \
    -c ba_config_rational.json 
        """)
        put_success('Done!')    
    

def on_global_landmarks_to_point3d():
    chdir_to_ballmove()
    import lilab.multiview_scripts.landmarks2point3d as mc
    mc.main(landmarks='../ballglobal/landmarks_global_pixel.json',
            poses = 'output/bundle_adjustment/ba_poses.json',
            out = 'global_points3d.json',
            filenames = 'filenames.json',
            dump_images = True, 
            setup = 'setup.json')
    with use_scope('ballglobal_messages', clear=True):
        put_text("""python -m lilab.multiview_scripts.landmarks2point3d \
    -l landmarks_global_pixel.json \
	-p output/bundle_adjustment/ba_poses.json \
	--out global_points3d.json \
    --dump_images \
	-s setup.json \
	-f filenames.json
        """)
        put_success('Done!')    

def on_global_registration():
    chdir_to_ballmove()
    import multiview_calib.scripts.global_registration as mc
    mc.main(setup = 'setup.json',
            ba_poses = 'output/bundle_adjustment/ba_poses.json',
            ba_points = 'global_points3d.json',
            landmarks = 'landmarks_global_pixel.json',
            landmarks_global = 'landmarks_global_world.json',
            filenames = 'filenames.json',
            dump_images = True)
    with use_scope('ballglobal_messages', clear=True):
        put_text("""python  -m multiview_calib.scripts.global_registration \
	-s setup.json \
	-ps output/bundle_adjustment/ba_poses.json \
	-po global_points3d.json \
	-l landmarks_global_pixel.json \
	-lg landmarks_global_world.json \
	-f filenames.json \
	--dump_images
        """)
        put_success('Done!')

def app(parent=None):
    if not parent:
        parent = 'ballmove'
        put_scope(parent)
    
    with use_scope(parent):
        pin.put_input('dlc_project_moveball', label='The Deeplabcut project folder')
        pin.put_input('move_ball_path', label='The path of moving ball')
        pin.put_input('start_time', label='The start time of ball', placeholder='00:00:01')
        
        put_button('1. crop video', on_crop_video)
        put_button('2. deeplabcut predict ball', on_deeplabcut_predict_ball)
        put_button('3. csv file to landmarkjson', on_csv_to_landmarkjson)
        put_button('3.5 copy configs to this folder', on_copy_configs)
        put_button('4. compute relative poses (robust)', on_compute_relative_poses)
        put_button('5. concatenate relative poses', on_concatenate_relative_poses)
        put_button('6. bundle adjustment', on_bundle_adjustment)
        put_button('7. global landmarsk to point3d', on_global_landmarks_to_point3d)
        put_button('8. global registration', on_global_registration)
        put_scope(scope_msg)
