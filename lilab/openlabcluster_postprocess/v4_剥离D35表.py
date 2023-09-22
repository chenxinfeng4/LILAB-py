# %%
import pickle
import pandas as pd
import numpy as np
import os.path as osp
usv_project  = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/DayAll20230828/usv_label/SexualDevelopD35D55D75_USV_recon'
usv_evt_file = osp.join(usv_project, 'usv_evt.usvpkl')
usv_latent_file = osp.join(usv_project, 'usv_latent.usvpkl')

#%%
usv_evt_data = pickle.load(open(usv_evt_file, 'rb'))
usv_latent_data = pickle.load(open(usv_latent_file, 'rb'))
assert tuple(usv_evt_data['video_nakes']) == tuple(usv_latent_data['video_nakes'])

# %%
video_nakes_pick = ['2023-07-24_13-58-48CwxDb',
'2023-07-24_14-19-10CbxDw',
'2023-07-24_14-37-09GwxHb',
'2023-07-24_14-56-29GbxHw',
'2023-07-25_14-00-26CwxHb',
'2023-07-25_14-20-01CbxHw',
'2023-07-25_14-38-46DbxGw',
'2023-07-25_14-58-31DwxGb',
'2023-07-26_14-14-04GwxHb',
'2023-07-26_14-33-39GbxHw',
'2023-07-26_14-52-07CwxDb',
'2023-07-26_15-11-08CbxDw',
'2023-07-27_13-56-43DbxHw',
'2023-07-27_14-17-11DwxHb',
'2023-07-27_14-37-10CwxGb',
'2023-07-27_14-56-29CbxGw']
video_nakes_pick.sort()

usv_latent = {v:usv_latent_data['usv_latent'][v] for v in video_nakes_pick}


df_usv_latent = pd.DataFrame(usv_latent_data['usv_latent'])