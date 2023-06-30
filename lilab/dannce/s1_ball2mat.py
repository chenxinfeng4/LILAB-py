# python -m lilab.dannce.s1_ball2mat xx.calibpkl

import pickle
import scipy.io as sio
import argparse
import os.path as osp
import numpy as np


pklfile = '/mnt/ftp.rat/multiview_large/lhj_20221014/errorframe/2022-10-13_14-42-58Sball.calibpkl'

def convert(pklfile):
    data = pickle.load(open(pklfile, 'rb'))
    ba_poses = [data['ba_poses'][i] for i in range(len(data['ba_poses']))]
    
    pklbytes = pickle.dumps({'ba_poses':data['ba_poses']})
    pklbytes_np = np.frombuffer(pklbytes, dtype=np.uint8)

    dataout = {'ba_poses': ba_poses, 'pklbytes': pklbytes_np}
    sio.savemat(pklfile+'.mat', dataout)

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pklfile', type=str)
    args = parser.parse_args()
    pklfile = args.pklfile
    assert osp.exists(pklfile), 'pklfile not exists'
    convert(pklfile)

# %%
