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
from lilab.multiview_scripts.webgui_ballglobal import app as app_ballglobal
from lilab.multiview_scripts.webgui_ballmove import app as app_ballmove
# /home/liying_lab/chenxinfeng/DATA/multiview-project/ratkeypoints_11-26/mulitview
# /home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_side6/ratBblack

def checkfolderexist():
    isok = True
    with use_scope('calib_check', clear=True):
        if not osp.exists(pin.pin.calib):
            put_error('Not exist!')
            isok = False
    with use_scope('folderrat_check', clear=True):
        if not osp.exists(pin.pin.folderrat):
            put_error('Not exist!')
            isok = False
    clear('messages')
    return isok
        
def on_dlc_to_landmarks():
    # check all need
    isok = checkfolderexist()
    if not isok: return False

    # check the csv file number
    folderrat = pin.pin.folderrat
    with use_scope('messages', clear=True):
        ncsv = len(glob.glob(osp.join(folderrat, '*.csv')))
        if ncsv == 0:
            put_error('No csv file in folder')
            isok = False
        elif ncsv != 6:
            put_error('The number of csv file is not 6, but get {} files'.format(ncsv))
            isok = False
    if not isok: return False

    # python -m lilab.multiview_scripts.dlcBall_2_landmarks $RAT --out $PROJECT_3D/landmarks_rat_pixel.json
    import lilab.multiview_scripts.dlcBall_2_landmarks
    outjson = osp.join(folderrat, 'landmarks_rat_pixel.json')
    if osp.isfile(outjson): os.remove(outjson)
    lilab.multiview_scripts.dlcBall_2_landmarks.main(folderrat, outjson)

    with use_scope('messages', clear=True):
        put_text('python -m lilab.multiview_scripts.dlcBall_2_landmarks . --out ./landmarks_rat_pixel.json')
        if osp.exists(outjson):
            put_success('Success!')
        else:
            put_error('Fail!')
        
def on_landmarks_to_3D_arbunit():
    # check all need
    isok = checkfolderexist()
    if not isok: return False

    # check the file existed
    folderrat = pin.pin.folderrat
    foldercalib = pin.pin.calib
    files_should_in_foldercalib = ['filenames.json', 'global_points3d.json', 'landmarks.json',
                                 'setup.json', 'landmarks_global_world.json', 'landmarks_global_pixel.json',
                                 'ba_config_rational.json', 'intrinsics_rational.json',
                                 'output/bundle_adjustment/ba_poses.json',
                                 'output/global_registration/global_poses.json']
    with use_scope('messages', clear=True):
        for f in files_should_in_foldercalib:
            if not osp.exists(osp.join(foldercalib, f)):
                put_error('Not exist {}'.format(f))
                isok = False
    if not isok: return False

    # python -m lilab.multiview_scripts.landmarks_to_3D $RAT --out $PROJECT_3D/landmarks_rat_pixel.json
    import lilab.multiview_scripts.landmarks2point3d
    califulp = lambda f: osp.join(foldercalib, f)
    ratfulp = lambda f: osp.join(folderrat, f)

    outjson = ratfulp('rat_points3d_arbunit.json')
    if osp.isfile(outjson): os.remove(outjson)
    lilab.multiview_scripts.landmarks2point3d.main(
            poses      = califulp('output/bundle_adjustment/ba_poses.json'),
            landmarks  = ratfulp('landmarks_rat_pixel.json'),
            out        = outjson,
            dump_images= True,
            setup      = califulp('setup.json'),
            filenames  = califulp('filenames.json'))

    # move folder: mv output/custom output/rat3d_arbunit
    if osp.exists(ratfulp('output/custom')):
        os.rename(ratfulp('output/custom'), ratfulp('output/rat3d_arbunit'))

    with use_scope('messages', clear=True):
        put_text('''python -m lilab.multiview_scripts.landmarks2point3d \
    -l landmarks_rat_pixel.json \
	-p $CALIB3D/output/bundle_adjustment/ba_poses.json \
	--out rat_points3d.json \
    --dump_images \
	-s $CALIB3D/setup.json \
	-f $CALIB3D/filenames.json
        ''')
        if osp.exists(outjson):
            put_success('Success!')
        else:
            put_error('Fail!')


def on_landmarks_to_3D_cm():
    # check all need
    isok = checkfolderexist()
    if not isok: return False

    # check the file existed
    folderrat = pin.pin.folderrat
    foldercalib = pin.pin.calib
    files_should_in_foldercalib = ['filenames.json', 'global_points3d.json', 'landmarks.json',
                                 'setup.json', 'landmarks_global_world.json', 'landmarks_global_pixel.json',
                                 'ba_config_rational.json', 'intrinsics_rational.json',
                                 'output/bundle_adjustment/ba_poses.json',
                                 'output/global_registration/global_poses.json']
    with use_scope('messages', clear=True):
        for f in files_should_in_foldercalib:
            if not osp.exists(osp.join(foldercalib, f)):
                put_error('Not exist {}'.format(f))
                isok = False
    if not isok: return False

    # python -m lilab.multiview_scripts.landmarks_to_3D $RAT --out $PROJECT_3D/landmarks_rat_pixel.json
    import lilab.multiview_scripts.landmarks2point3d
    califulp = lambda f: osp.join(foldercalib, f)
    ratfulp = lambda f: osp.join(folderrat, f)

    outjson = ratfulp('rat_points3d_cm.json')
    if osp.isfile(outjson): os.remove(outjson)
    lilab.multiview_scripts.landmarks2point3d.main(
            poses      = califulp('output/global_registration/global_poses.json'),
            landmarks  = ratfulp('landmarks_rat_pixel.json'),
            out        = outjson,
            dump_images= False,
            setup      = califulp('setup.json'),
            filenames  = califulp('filenames.json'))

    # move folder: mv output/custom output/rat3d_arbunit
    if osp.exists(ratfulp('output/custom')):
        os.rename(ratfulp('output/custom'), ratfulp('output/rat3d_cm'))

    with use_scope('messages', clear=True):
        put_text('''python -m lilab.multiview_scripts.landmarks2point3d \
    -l landmarks_rat_pixel.json \
	-p $CALIB3D/output/global_registration/global_poses.json \
	--out rat_points3d_cm.json
        ''')
        if osp.exists(outjson):
            put_success('Success!')
        else:
            put_error('Fail!')


def on_3D_cm_to_matlab():
    # check all need
    isok = checkfolderexist()
    if not isok: return False
    json_3D_cm = osp.join(pin.pin.folderrat, 'rat_points3d_cm.json')
    
    out_file = json_3D_cm.replace('.json', '.mat')
    if osp.isfile(out_file): os.remove(out_file)
    
    import lilab.multiview_scripts.ratpoint3d_to_mat
    lilab.multiview_scripts.ratpoint3d_to_mat.convert(json_3D_cm)
    
    with use_scope('messages', clear=True):
        put_text('python -m lilab.multiview_scripts.ratpoint3d_to_mat {}'.format(json_3D_cm))
        if osp.exists(out_file):
            put_success('Success!')
        else:
            put_error('Fail!')
            

def on_matlab_to_video():
    # check all need
    isok = checkfolderexist()
    if not isok: return False
    mat_3D_cm = osp.join(pin.pin.folderrat, 'rat_points3d_cm.mat')

    out_file = osp.join(pin.pin.folderrat, 'rat3dskeleton.mp4')
    if osp.isfile(out_file): os.remove(out_file)

    import lilab.multiview_scripts.ratpoint3d_to_video
    lilab.multiview_scripts.ratpoint3d_to_video.main_plot3d(mat_3D_cm, None)
    
    with use_scope('messages', clear=True):
        put_text('python -m lilab.multiview_scripts.ratpoint3d_to_video xxxx')
        if osp.exists(out_file):
            put_success('Success!')
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


def app():
    sc_rat = put_scope('outrat')
    sc_tworat = put_scope('outratcheck3')
    sc_ballglobal = put_scope('ballglobal')
    sc_ballmove = put_scope('ballmove')
    put_tabs([
    {'title': 'Ball global', 'content': sc_ballglobal},
    {'title': 'Ball move', 'content': sc_ballmove},
    {'title': "Rat", 'content': sc_rat},
    {'title': "Two Rats", 'content': sc_tworat},
    ])
    app_ballglobal('ballglobal')
    app_ballmove('ballmove')

    with use_scope('outrat'):
        pin.put_input('calib', label = 'The calibration folder', placeholder='/home/calib')
        put_scope('calib_check')
        pin.put_input('folderrat', label = 'The rat video folder', placeholder= '/home/rat') 
        put_scope('folderrat_check')
        put_button('1.dlc to landmarks',  onclick=on_dlc_to_landmarks)
        put_button('2.landmarks to 3D (arbitrary unit, optinal)',  onclick=on_landmarks_to_3D_arbunit)
        put_button('3.landmarks to 3D (cm unit)',  onclick=on_landmarks_to_3D_cm)
        put_button('4. 3D (cm unit) to matlab',  onclick=on_3D_cm_to_matlab)
        put_button('5. 3D (cm unit)/matlab to video',  onclick=on_matlab_to_video)
        put_scope('messages')

if __name__ == '__main__':
    start_server(app, debug=True, port='44315')
