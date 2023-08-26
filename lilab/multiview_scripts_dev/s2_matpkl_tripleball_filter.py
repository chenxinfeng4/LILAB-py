# python -m lilab.multiview_scripts_dev.s2_matpkl——tripleball_filter xxx.matpkl --iframe x --iview x
import argparse
import pickle 
import numpy as np
import os.path as osp
#错误点帧数iframe，视角iview
pklfile='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zzc_to_ouyang/2302oxtr3dq-m/2023-02-22_15-12-37_ballnishizhen.matpkl'


#%%错误点的坐标
def main(pklfile:str, ikpt:int, iframe:int, iview:int):
    with open(pklfile, 'rb') as f:
        pkldata = pickle.load(f)
        
    nview, nframe,nnumber, ncoord = pkldata['keypoints'].shape #(nview, nframe, 1,3)
    assert iframe<nframe and iview<nview and ncoord==3 and nnumber>ikpt

    trace = pkldata['keypoints'][iview, iframe, ikpt]
    distance = np.squeeze(np.linalg.norm(pkldata['keypoints'][iview,:,ikpt] - trace, axis=-1))
    iframe_errors = distance < 10
    n_error = iframe_errors.sum()
    if n_error:
        print('%d outliers removal'%(n_error))
    pkldata['keypoints'][iview, iframe_errors] = [[np.nan, np.nan, 0]]

    with open(pklfile, 'wb') as f:
        pickle.dump(pkldata, f)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pklfile', type=str)
    parser.add_argument('ikpt', type=int)
    parser.add_argument('--iframe', type=int)
    parser.add_argument('--iview', type=int)
    

    args = parser.parse_args()
    assert osp.isfile(args.pklfile)

    main(args.pklfile, args.ikpt, args.iframe, args.iview)
