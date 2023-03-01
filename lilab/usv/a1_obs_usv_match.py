# %%
import os
import os.path as osp
import glob
import datetime
from typing import Union

obsfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/SOLO/2023-01-02_15-06-07DwWT_mask.mp4'
USV_folders = ['/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/USV']

# %%
def get_usv_file(obsfile, USV_files):
    obs_datetimestr = osp.basename(obsfile)[:19]
    obs_datetime = datetime.datetime.strptime(obs_datetimestr, "%Y-%m-%d_%H-%M-%S")
    usv_datetime_candidate = [obs_datetime+datetime.timedelta(seconds=s) for s in [-1,0,1]]

    usv_datetimestr_candidate = [datetime.datetime.strftime(u,"%Y-%m-%d_%H-%M-%S") 
                                    for u in usv_datetime_candidate]

    usv_files = []
    for uf in USV_files:
        for usv_datetimestr in usv_datetimestr_candidate:
            if usv_datetimestr in uf:
                usv_files.append(uf)
                break
    
    if len(usv_files)>=2:
        raise 'multiple match'+str(usv_files)
    elif len(usv_files)==0:
        raise 'empty match'+str(obs_datetimestr)

    usv_file = usv_files[0]
    return usv_file


def get_usv_file_byfolder(obsfiles:Union[str,list], USV_folders:Union[str,list], ext='.WAV') -> Union[list, bool]:
    if isinstance(USV_folders,str):
        USV_folders = [USV_folders]

    obsfilesl = [obsfiles] if isinstance(obsfiles, str) else obsfiles
    assert all(osp.isdir(folder) for folder in USV_folders)
    USV_files_all = sum([glob.glob(osp.join(folder, f'*{ext}')) for folder in USV_folders], [])
    usv_filesl = [get_usv_file(obsfile, USV_files_all) for obsfile in obsfilesl]

    usv_files = usv_filesl[0] if isinstance(obsfiles, str) else usv_filesl
    return usv_files

