# python -m lilab.multiview_scripts_dev.s3_ballpkl_fliter xxx.ballpkl
# %%
import argparse
import pickle
import numpy as np
import os.path as osp

#pklfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zzc_to_ouyang/2302oxtr3dq-m/2023-02-22_15-12-37_ballnishizhen.calibpkl'
pklfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zzc_to_ouyang/2302oxtr3dq-m/2023-02-22_15-12-37_ballnishizhen.calibpkl'

#pklfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zzc_to_dzy/20230218_oxtr-c/2023-02-22_15-12-37_ballnishizhen.calibpkl'
#%%
def main(pklfile):
    with open(pklfile, 'rb') as f:
        pkldata = pickle.load(f)

    nview, nframe, ncoord = pkldata['landmarks_move_xy'].shape #(nview, nframe, 2)
    assert nframe>100, 'nframe must > 100'
    assert ncoord==2, 'ncoord must == 2'
    modified = False

    for iview in range(nview):
        trace = pkldata['landmarks_move_xy'][iview]
        ind_notnan = np.isnan(trace[:,0])==False
        trace_clean = trace.copy()[ind_notnan]
        H, xedges, yedges = np.histogram2d(trace_clean[:,0], trace_clean[:,1], bins=100)

        # find the peak
        ind = np.unravel_index(np.argmax(H, axis=None), H.shape)
        vpeak = H[ind[0], ind[1]]
        ind = [np.clip(ind[0], 1, H.shape[0]-2), np.clip(ind[1], 1, H.shape[1]-2)]

        #  
        Hnext = H.copy()
        Hnext[ind[0]-1:ind[0]+2, ind[1]-1:ind[1]+2] = 0
        indnext = np.unravel_index(np.argmax(Hnext, axis=None), Hnext.shape)
        vpeaknext = Hnext[indnext[0], indnext[1]]

        if vpeak / vpeaknext < 5: continue   
        x_range = xedges[[ind[0]-1,ind[0]+2]]
        y_range = yedges[[ind[1]-1,ind[1]+2]]
        ind_outlier =  ((trace[:,0]> x_range[0]) & (trace[:,0]<x_range[1]) &
                        (trace[:,1]> y_range[0]) & (trace[:,1]<y_range[1]))

        pkldata['landmarks_move_xy'][iview, ind_outlier, :] = np.nan
        print('view %d: %d outliers removal'%(iview, np.sum(ind_outlier)))
        modified = True

    # %% inplace save pickle
    if modified:
        with open(pklfile, 'wb') as f:
            pickle.dump(pkldata, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pklfile', type=str)
    args = parser.parse_args()

    pklfile = args.pklfile
    assert osp.exists(pklfile), 'pklfile not exists'
    main(pklfile)
