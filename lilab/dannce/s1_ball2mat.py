# python -m lilab.dannce.s1_ball2mat xx.calibpkl
import pickle
import scipy.io as sio
import argparse
import os.path as osp

pklfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/2021-11-02-side6-bwrat_800x600red/ball/ball.calibpkl'
def convert(pklfile):
    data = pickle.load(open(pklfile, 'rb'))
    ba_poses = [data['ba_poses'][i] for i in range(len(data['ba_poses']))]
    dataout = {'ba_poses': ba_poses}
    sio.savemat(pklfile+'.mat', dataout)

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pklfile', type=str)
    args = parser.parse_args()
    pklfile = args.pklfile
    assert osp.exists(pklfile), 'pklfile not exists'
    convert(pklfile)
