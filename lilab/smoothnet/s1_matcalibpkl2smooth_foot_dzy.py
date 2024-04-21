# conda activate mmpose
# python -m lilab.smoothnet.s1_matcalibpkl2smooth_foot_dzy test.matcalibpkl
# %% imports
import pickle
import mmpose
from mmpose.core.post_processing.temporal_filters import build_filter
from lilab.multiview_scripts_dev.s6_calibpkl_predict import CalibPredict
import numpy as np
import argparse
from scipy.signal import medfilt
import itertools

# %%
pklfile='/home/liying_lab/chenxinfeng/liying.cibr.ac.cn_Data_Temp/multiview_9/dzytemp/2023-03-20_14-35-51_MA3B1.matcalibpkl'

checkpoint='/home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/smoothnet/checkpoint/smoothnet_ws64_cxf.pth'
checkpoint_foot='/home/liying_lab/chenxinfeng/DATA/SmoothNet/results/h36m_cxf_16b3/checkpoint.pth.tar'
# checkpoint_foot='https://download.openmmlab.com/mmpose/plugin/smoothnet/smoothnet_ws16_h36m.pth'

def filter_xyz(X, filter):
    T, N, K, D = X.shape
    X_re = X.reshape(T, N*K, D)
    X_smoothed = filter(X_re)
    X_smoothed = X_smoothed.reshape(T, N, K, D)
    return X_smoothed

def linear_smooth(data, window_size=3):
    """
    对输入的运动数据进行线性平滑
    每一个时间点上的值取前后N帧的中位数
    """
    smoothed_data = data.copy()
    ntime, nclass, npoints, nspace = data.shape
    for iclass, ispace, ipoint in itertools.product(range(nclass), range(nspace), range(npoints)):
        smoothed_data[:, iclass, ipoint, ispace] = medfilt(data[:, iclass, ipoint, ispace], kernel_size=window_size*2+1)
    return smoothed_data

def main(pklfile):
    filter_cfg = dict(
        type='SmoothNetFilter',
        window_size=64,
        output_size=64,
        checkpoint=checkpoint,
        hidden_size=512,
        res_hidden_size=256,
        num_blocks=3,
        root_index=None)
    filter = build_filter(filter_cfg)
    # 脚部的filter
    filter_cfg_foot = dict(
        type='SmoothNetFilter',
        window_size=16,
        output_size=16,
        checkpoint=checkpoint_foot,
        hidden_size=512,
        res_hidden_size=256,
        num_blocks=3,
        root_index=None)
    filter_foot = build_filter(filter_cfg_foot)

    # %%
    pkldata=pickle.load(open(pklfile, 'rb'))
    calibPredict=CalibPredict(pkldata)
    X_global = pkldata['keypoints_xyz_ba'].astype(float)
    coms_3d = pkldata['extra']['coms_3d'][:, :, None, :]
    p_max = pkldata['extra']['keypoints_xyz_pmax']
    X_centored = X_global - coms_3d
    X_centored = X_centored[:].astype(np.float32)
    centor = coms_3d[:].astype(np.float32)

    # 鼠的身体长度
    ntime, nclass, npoints, nspace = X_centored.shape
    body_length = np.percentile(np.linalg.norm(X_global[:,:,0,:] - X_global[:,:,5], axis=-1), 95, axis=0)
    # 设定不同部位的threshold
    thresholdlist = np.zeros((npoints, nclass))
    slow_threshold = [x/50 for x in body_length]
    mid_threshold = [x/20 for x in body_length]
    fast_threshold = [x/5 for x in body_length]
    # 三层smooth，首先是慢速部分，主要是脖子后背
    slow_KPT_index = [3,4]
    # 中速部份的关键点，主要是肩膝和尾部
    mid_KPT_index = [5,6,8,10,12]
    # 快速部分，主要是脚部和头部的关键点
    fast_KPT_index = [0,1,2,7,9,11,13]
    # 放进thresholdlist
    thresholdlist[slow_KPT_index,:] = slow_threshold
    thresholdlist[mid_KPT_index,:] = mid_threshold
    thresholdlist[fast_KPT_index,:] = fast_threshold
    thresholdlist = thresholdlist.T
    # 先做一遍smooth，针对foot做一遍timewindow更短的smooth
    foot_KPT_index = [7,9,11,13]
    X_smoothed = filter_xyz(X_centored, filter)
    X_smoothed_foot = filter_xyz(np.ascontiguousarray(X_centored[:,:,foot_KPT_index,:]), filter_foot)
    X_smoothed[:,:,foot_KPT_index,:] = X_smoothed_foot

    # 用threshold过滤X_smoothed和X_centored,并放进X_merge
    X_merge = np.where((np.linalg.norm(X_smoothed - X_centored, axis=3) > thresholdlist)[:, :, :,None], X_smoothed, X_centored)
    
    # 再做一遍中值滤波
    X_smoothed_final = linear_smooth(X_merge, window_size=2)

    X_global_smoothed = X_smoothed_final + centor

    X_global_smoothed_xy = calibPredict.p3d_to_p2d(X_global_smoothed)
    outdict = {**pkldata, 'keypoints_xyz_ba': X_global_smoothed.astype(np.float16),
                'keypoints_xy_ba': X_global_smoothed_xy.astype(np.float16)}

    outpklfile =  pklfile.replace('.matcalibpkl', '.smoothed_foot.matcalibpkl')
    pickle.dump(outdict, open(outpklfile, 'wb'))
    print('Saved to', outpklfile)
    # print('Generating videos...')
    # from lilab.mmpose.s3_matcalibpkl_2_video2d import main
    # main(outpklfile, 1, 'smoothed_foot')
    return outpklfile

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pklfile', type=str)
    args = parser.parse_args()
    outpklfile = main(args.pklfile)