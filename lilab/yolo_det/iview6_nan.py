#%%
#python -m lilab.yolo_det.iview6_nan $matpkl
import pickle
import numpy as np
import argparse
#calibpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/ball/2022-04-29_17-58-45_ball.calibpkl'
matpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/marmoset_camera3_cxf/2024-2-21_marmosettracking/2024-02-21_15-07-36.matpkl'
def convert(matpkl):  
    data=pickle.load(open(matpkl, "rb"))
    data['keypoints'][5,...]=np.nan
    pickle.dump(data, open(matpkl, "wb"))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('matpkl', type=str)
    args = parser.parse_args()
    convert(args.matpkl)