# python -m lilab.bea_wpf.s2_create_bea_project A/B/C/BeA_WPF
"""
#%%
├── data (无所谓，源数据)
│   ├── rec-1-malexxxlicl-20230308164621-caliParas.mat
│   ├── rec-1-malexxxlicl-20230308164621-camera-0.avi
│   ├── rec-1-malexxxlicl-20230308164621-camera-1.avi
│   ├── rec-1-malexxxlicl-20230308164621-camera-2.avi
│   ├── rec-1-malexxxlicl-20230308164621-camera-3.avi
├── PreVideo (无所谓，STEP1 的中间结果)
│   ├── BlackMouse_WhiteBackground.avi
│   ├── BlackMouse_WhiteBackgroundDLC_resnet50_black_mice720Jul28shuffle1_1030000.csv
│   ├── BlackMouse_WhiteBackgroundDLC_resnet50_black_mice720Jul28shuffle1_1030000.h5
│   └── BlackMouse_WhiteBackgroundDLC_resnet50_black_mice720Jul28shuffle1_1030000_meta.pickle
├── ProjectConfig.json
└── results
    ├── 3Dskeleton （STEP2 的结果）
    │   ├── rec-1-malexxxlicl-20230308164621_Cali_Data3d.c3d
    │   ├── rec-1-malexxxlicl-20230308164621_Cali_Data3d.csv
    ├── BeAOutputs
    │   ├── rec-1-malexxxlicl-20230308164621_results.h5
    ├── Behavior_Atlas.json
    ├── figures
    │   ├── clustergram.fig
    │   ├── clustergram.jpg
    │   ├── feat_space_scatter3D.fig
    │   ├── feat_space_scatter3D.jpg
    │   ├── rec-1-malexxxlicl-20230308164621_skl_view.jpg
    ├── tree.mat
    └── VideoSeg
"""

# %% imports
import json
import glob
import os
import os.path as osp
from lilab.bea_wpf.s1_matcalibpkl_to_bea_3d import video_name_to_bea_name
import shutil
import glob
import tqdm
import argparse
import ffmpegcv

Behavior_Atlas_template = '/home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/bea_wpf/templates/Behavior_Atlas.json'
Behavior_Atlas = json.load(open(Behavior_Atlas_template, 'rb'))
projectdir='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zyq_to_dzy/20221105_1/multi/BeA_WPF'

#%%
def create_Behavior_Atlas_json(projectdir:str):
    resultdir=osp.join(projectdir, 'results')
    infiles = glob.glob(osp.join(resultdir, 'BeAOutputs', '*.h5'))
    assert len(infiles)
    proj_file_l = [osp.basename(f).replace('_results.h5', '') for f in infiles]
    # assert all('Calib' not in f for f in proj_file_l)

    # Behavior_Atlas['group_fileLists']['UnGrouped'] = proj_file_l

    # Behavior_Atlas_file = osp.join(resultdir, 'Behavior_Atlas.json')
    # with open(Behavior_Atlas_file, 'w') as f:
    #     json.dump(Behavior_Atlas, f, indent=4)

    return proj_file_l


def create_ProjectConfig_json(projectdir:str, proj_file_l:list):
    ProjectConfig=dict()
    ProjectConfig['projectName']=proj_file_l
    ProjectConfig['last_model']="BlackMouse_BlackBackground"
    ProjectConfig['step_1']={f:1 for f in proj_file_l}
    ProjectConfig['step_2']={f:0 for f in proj_file_l}
    ProjectConfig['step_3']={f:0 for f in proj_file_l}
    ProjectConfig['step_4']={f:0 for f in proj_file_l}
    ProjectConfig_file = osp.join(projectdir, 'ProjectConfig.json')
    with open(ProjectConfig_file, 'w') as f:
        json.dump(ProjectConfig, f, indent=4)


def create_empty_dlc_result(datadir:str, proj_file_l:list):
    outvfile_l = []
    for bea_nakename in proj_file_l:
        outvfile_l.extend([osp.join(datadir, bea_nakename + '-camera-%d' % i) for i in range(4)])

    post_l = ['.csv', '.h5', '_meta.pickle']
    for outvfile in outvfile_l:
        for post in post_l:
            fullfile_empty = outvfile + post
            open(fullfile_empty, 'w').close()


def create_fake_caliParas(datadir, proj_file_l):
    targets = [osp.join(datadir, f+'-caliParas.mat') for f in proj_file_l]
    for target in targets:
        open(target, 'w').close()


def create_video_4views(datadir:str, v_name_as_project:dict):
    for nake_project, vfile in tqdm.tqdm(v_name_as_project.items()):
        outvfile_l = [osp.join(datadir, nake_project + '-camera-%d.avi' % i) for i in range(4)]
        videoinfo = ffmpegcv.video_info.get_info(vfile)
        assert videoinfo.codec in ['hevc', 'h264'], f'video codec not supported {videoinfo["codec"]}'
        codec = videoinfo.codec + '_cuvid'
        ori_wh = (videoinfo.width, videoinfo.height)
        if ori_wh == (1280*3, 800*3):
            bottom=800
            right=1280
        elif ori_wh == (1280*2, 800*2):
            bottom=0
            right=0
        elif ori_wh == (1280*3, 800*2):
            bottom=0
            right=1280
        else:
            raise ValueError('video size not supported')
        top=0
        left=0
        dst_w=1280//2
        det_h=800//2
        re_width = dst_w*2
        re_height = det_h*2

        cmd = (f'ffmpeg -loglevel warning -hwaccel cuda -hwaccel_device 0 -c:v {codec}'
        f' -resize {re_width}x{re_height}'
        f' -crop {top}x{bottom}x{left}x{right}'
        f' -i "{vfile}"'
        f' -filter_complex "[0:v]crop={dst_w}:{det_h}:0:0[v0];[0:v]crop={dst_w}:{det_h}:{dst_w}:0[v1];[0:v]crop={dst_w}:{det_h}:0:{det_h}[v2];[0:v]crop={dst_w}:{det_h}:{dst_w}:{det_h}[v3]"'
        f' -map "[v0]" -y "{outvfile_l[0]}"'
        f' -map "[v1]" -y "{outvfile_l[1]}"'
        f' -map "[v2]" -y "{outvfile_l[2]}"'
        f' -map "[v3]" -y "{outvfile_l[3]}"'
        )

        print(cmd)
        os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Create BeA project from the BeA_WPF dir.')
    parser.add_argument('bea_dir', type=str)
    args = parser.parse_args()

    # make dirs
    projectdir = args.bea_dir
    assert osp.isdir(projectdir) and  osp.isdir(osp.join(projectdir, 'results'))
    datadir=osp.join(projectdir, 'data')
    os.makedirs(datadir, exist_ok=True)

    # find the original videos
    vfiles = glob.glob(osp.join(osp.dirname(projectdir), '*.mp4'))
    vfiles = [v for v in vfiles if ('sktdraw' not in v  and
                                    'com3d' not in v and
                                    'mask' not in v and                                 
                                    'vol210' not in v and
                                    '400p' not in v)]
    v_name_as_project = {video_name_to_bea_name(v): v for v in vfiles}
    assert len(v_name_as_project) == len(vfiles)

    # create the files
    proj_file_l = create_Behavior_Atlas_json(projectdir)
    assert len(proj_file_l)>0 and set(proj_file_l) <= set(v_name_as_project)
    v_name_as_project_join = {f:v_name_as_project[f] for f in proj_file_l}
    # create the project
    create_ProjectConfig_json(projectdir, proj_file_l)
    create_empty_dlc_result(datadir, proj_file_l)
    create_fake_caliParas(datadir, proj_file_l)
    create_video_4views(datadir, v_name_as_project_join)

# %%
