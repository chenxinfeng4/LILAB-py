# python -m lilab.multiview_scripts_dev.s3_ballpkl_fliter xxx.ballpkl
# %%
import argparse
import pickle 
import numpy as np
import os.path as osp
#错误点帧数iframe，视角iview
pklfile='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zzc_to_ouyang/2302oxtr3dq-m/2023-02-22_15-12-37_ballnishizhen.matpkl'


#%%错误点的坐标
def main(pklfile):
    with open(pklfile, 'rb') as f:
        pkldata = pickle.load(f)
        
    nview, nframe,nnumber, ncoord = pkldata['keypoints'].shape #(nview, nframe, 1,3)
    trace = pkldata['keypoints'][iframe][iview]
    errorpoint_x = np.squeeze(trace[0][...,0])
    errorpoint_y = np.squeeze(trace[0][...,1])
    x = np.squeeze(pkldata['keypoints'][7][...,0])
    y = np.squeeze(pkldata['keypoints'][7][...,1])
    for i in range(0,nframe,1):
        r=pow(pow(x[i]-errorpoint_x,2)+pow(y[i]-errorpoint_y,2),0.5)
        if r < 5:
            pkldata['keypoints'][7][...,0:2][i] = np.nan
            print('%d outliers removal'%(i))

#%%
with open(pklfile, 'wb') as f:
    pickle.dump(pkldata, f)
    



##错误点帧数和视角：
# iview = 7
# iframe = 1243






# %%
