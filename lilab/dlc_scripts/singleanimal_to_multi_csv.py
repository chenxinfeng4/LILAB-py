# %%
import os
import os.path as osp
import pandas as pd
import numpy as np

source = '/home/liying_lab/chenxinfeng/deeplabcut-project/bwrat_28kpt-cxf-2022-02-25/labeled-data/2021-11-02-bwrat_side6-kp_trainval/CollectedData_cxf.csv'
target_folder = '/home/liying_lab/chenxinfeng/deeplabcut-project/bwrat_14ratkptMulti-cxf-2022-02-26/labeled-data/15-11-45_output_1_15fps_2min_black/'

source_h5 = osp.join(osp.dirname(source), 'CollectedData_cxf.h5')
target_h5 = osp.join(target_folder, 'CollectedData_cxf.h5')
target_csv = osp.join(target_folder, 'CollectedData_cxf.csv')

# %% load data
df = pd.read_hdf(source_h5, 'df_with_missing')
# %%

scorer='cxf'
individuals = ['rat_black', 'rat_white']
bodyparts = ['Nose', 'EarL',  'EarR', 'Neck', 'Back', 'Tail', 
             'ForeShoulderL', 'ForePowL', 'ForeShoulderR', 'ForePowR',
             'BackShoulderL', 'BackPowL', 'BackShoulderR', 'BackPowR']
coords = ['x', 'y']
header = pd.MultiIndex.from_product([[scorer], individuals, bodyparts, coords], 
                                    names=['scorer', 'individuals', 'bodyparts', 'coords'])
df.columns = header

# %%
df.to_csv(target_csv)
df.to_hdf(target_h5, 'df_with_missing')