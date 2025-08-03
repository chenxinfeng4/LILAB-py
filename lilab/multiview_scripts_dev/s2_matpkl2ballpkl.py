# python -m lilab.multiview_scripts_dev.s2_matpkl2ballpkl ../data/matpkl/ball.matpkl --time 1 12 24 35 48
import pickle
import numpy as np
import argparse
import os.path as osp
import cv2
from lilab.cameras_setup import get_ballglobal_cm, get_json_wrapper

second_based = True
pthr = 0.7 #0.7
# pthr = 0.62
nchoose = 1000
matfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/ball/2022-04-29_17-58-45_ball.matpkl'


def load_mat(matfile):
    # 从matfile中加载数据
    data = pickle.load(open(matfile, 'rb'))
    # 复制关键点数据
    keypoint = data['keypoints'].copy()
    # 获取帧率
    fps = data['info']['fps']
    # 获取视频文件名
    vfile = data['info']['vfile']
    # 获取视图的xywh信息
    views_xywh = data['views_xywh']

    # 获取关键点的类别数
    nclass = keypoint.shape[2]
    # 断言关键点的维度为4，最后一维为3
    assert keypoint.ndim == 4 and keypoint.shape[-1] == 3, "xyp is expected"
    # 如果类别数为1，则将关键点数据转换为二维数组
    if nclass==1:
        keypoint = keypoint[:,:,0,:]
    # 否则，将关键点数据按照类别数进行拼接
    else:
        keypoint = np.concatenate([keypoint[:,:,i,:] for i in range(nclass)], axis=1)
    # 返回关键点数据、帧率、视频文件名和视图的xywh信息
    return keypoint, fps, vfile, views_xywh


def auto_find_thr(pvals):
    # 将pvals展开成一维数组
    pvals_ravel = pvals.ravel()
    # 去除pvals_ravel中的NaN值
    pvals_ravel = pvals_ravel[~np.isnan(pvals_ravel)]
    # 计算pvals_ravel的第50百分位数，并乘以0.9
    pthr = np.percentile(pvals_ravel, 50) * 0.9
    # 返回pthr
    return pthr

def split_keypoint(keypoint, fps, global_time):

    # 将关键点的z坐标提取出来
    keypoint_p = keypoint[...,2]
    # 将关键点的x,y坐标提取出来
    keypoint_xy = keypoint[...,:2]  # VIEWxTIMExXY
    # 自动找到关键点的阈值
    pthr = auto_find_thr(keypoint_p)
    # 将关键点的x,y坐标中z坐标小于阈值的点置为nan
    keypoint_xy[keypoint_p < pthr] = np.nan
    # 移动时间
    move_time = global_time[-1] + 3
    # 如果是按照秒来计算
    if second_based:
        # 将全局时间乘以帧率，转换为整数
        global_index = np.array(np.array(global_time)*fps, dtype=int)
        # 将移动时间乘以帧率，转换为整数
        move_index = int(move_time*fps)
    else:
        # 将全局时间转换为整数
        global_index = np.array(global_time, dtype=int)
        # 将移动时间转换为整数
        move_index = int(move_time)
    # 提取全局时间的关键点坐标
    keypoint_xy_global = keypoint_xy[:, global_index, :]      # VIEWxTIMExXY
    # 提取移动时间的关键点坐标
    keypoint_xy_move = keypoint_xy[:, move_index:(-5*30), :]  # VIEWxTIMExXY
    # 返回全局时间的关键点坐标，移动时间的关键点坐标，全局时间
    return keypoint_xy_global, keypoint_xy_move, global_index


def get_background_img(global_iframe, vfile, views_xywh):
    # 定义一个空列表，用于存储背景图像
    img_stack = []
    # 定义一个空列表，用于存储裁剪后的背景图像
    background_img = []
    # 判断视频文件是否存在
    if osp.exists(vfile):
        # 打开视频文件
        vin = cv2.VideoCapture(vfile)
        # 遍历全局帧数
        for i in global_iframe:
            # 设置视频帧数
            vin.set(cv2.CAP_PROP_POS_FRAMES, i)
            # 读取视频帧
            ret, img = vin.read()
            # 断言是否成功读取帧
            assert ret, "Failed to read frame {}".format(i)
            # 将读取的帧添加到img_stack列表中
            img_stack.append(img)
        # 释放视频文件
        vin.release()
        # 将img_stack列表中的帧堆叠成三维数组
        img_stack = np.stack(img_stack, axis=0)
        # 计算背景画布，取img_stack列表中的帧的中位数
        background_canvas = np.median(img_stack, axis=0).astype(np.uint8) #HxWx3
        # 遍历裁剪区域
        for crop_xywh in views_xywh:
            # 获取裁剪区域的坐标和尺寸
            x, y, w, h = crop_xywh
            # 将背景画布中对应区域的图像添加到background_img列表中
            background_img.append(background_canvas[y:y+h, x:x+w])
    # 如果视频文件不存在
    else:
        # 遍历裁剪区域
        for crop_xywh in views_xywh:
            # 获取裁剪区域的坐标和尺寸
            x, y, w, h = crop_xywh
            # 将裁剪区域中的图像添加到background_img列表中，图像为全零
            background_img.append(np.zeros((h, w, 3), dtype=np.uint8))
    # 返回background_img列表
    return background_img


def downsampe_keypoint(keypoint_xy_move):
    # 获取非nan值的索引
    ind_notnan = ~np.isnan(keypoint_xy_move[:,:,0]) #(nview, nframe)
    # 获取至少有nview//2个非nan值的帧的索引
    ind_3notnan = np.sum(ind_notnan, axis=0) >= keypoint_xy_move.shape[0]//2

    # 获取帧数
    nframe = keypoint_xy_move.shape[1]
    # 根据ind_3notnan筛选出非nan值的帧
    keypoint_xy_move = keypoint_xy_move[:,ind_3notnan]
    # 获取筛选后的非nan值的索引
    ind_notnan = np.isnan(keypoint_xy_move[:,:,0]) #(nview, nframe)

    # 设置采样方式
    mycase=1
    # 如果采样方式为0或者帧数小于nchoose，则直接返回keypoint_xy_move
    if mycase==0 or nframe<nchoose:
        keypoint_xy_move_downsample = keypoint_xy_move
    # 如果采样方式为1，则每隔3帧采样一次
    elif mycase==1:
        keypoint_xy_move_downsample = keypoint_xy_move[:,::3,:]
    # 如果采样方式为2，则只保留所有视图中都有非nan值的帧
    elif mycase==2:
        ind_notnan_iframe = np.all(ind_notnan, axis=0) #(nframe)
        keypoint_xy_move_downsample = keypoint_xy_move[:, ind_notnan_iframe, :]
    # 如果采样方式为3，则根据非nan值的数量随机采样
    elif mycase==3:
        count_notnan_iframe = np.mean(ind_notnan, axis=0) #(nframe)
        p_iframe = np.clip(count_notnan_iframe, 0.3, 1)
        p_iframe = (x-np.min(p_iframe))/(np.max(p_iframe)-np.min(p_iframe))
        ind_rand = np.random.random(count_notnan_iframe.shape) < p_iframe
        keypoint_xy_move_downsample = keypoint_xy_move[:, ind_rand, :]
    # 获取采样后的视点数和帧数
    nview, nget=keypoint_xy_move_downsample.shape[:2]

    # 如果采样后的帧数大于nchoose，则随机选择nchoose个帧
    if nget>nchoose:
        ind_down = np.random.choice(nget, nchoose, replace=False)
        keypoint_xy_move_downsample = keypoint_xy_move_downsample[:, ind_down, :]
    # 返回采样后的关键点
    return keypoint_xy_move_downsample


def convert(matfile, global_time, force_setupname:str):
    # 加载mat文件
    keypoint, fps, vfile, views_xywh = load_mat(matfile)
    # 将关键点按照全局时间和帧率进行分割
    keypoint_xy_global, keypoint_xy_move, global_index = split_keypoint(keypoint, fps, global_time)
    # 对移动的关键点进行下采样
    keypoint_xy_move_downsample = downsampe_keypoint(keypoint_xy_move)
    # 获取背景图像
    background_img = get_background_img(global_index, vfile, views_xywh)
    # 获取全局坐标下的球体中心
    fitball_xyz_global= get_ballglobal_cm()
    # 获取setup和intrinsics的json文件
    setup_json, intrinsics_json = get_json_wrapper(force_setupname)
    
    # 断言intrinsics的json文件和views_xywh的长度相等
    assert len(intrinsics_json) == len(views_xywh)
    
    # 构造输出字典
    outdict = {'landmarks_global_xy': keypoint_xy_global,         # VIEWxTIMExXY
               'landmarks_move_xy': keypoint_xy_move_downsample,  # VIEWxTIMExXY
               'global_iframe': global_index,                    # TIME
               'landmarks_global_cm':  fitball_xyz_global,        # TIMExXYZ
               'background_img': background_img,                 # VIEWxHxWx3
               'setup': setup_json,                         # setup.json content
               'intrinsics': intrinsics_json,               # intrinsics.json content
               }
    # 生成输出文件名
    outfile = osp.splitext(matfile)[0] + '.ballpkl'
    # 将输出字典保存为pickle文件
    pickle.dump(outdict, open(outfile, 'wb'))
    # 打印输出文件名
    print('python -m lilab.multiview_scripts_dev.s3_ballpkl2calibpkl',
            "{}".format(outfile))
    # 返回输出文件名
    return outfile

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('matfile', type=str)
    parser.add_argument('--time', type=float, nargs='+')
    parser.add_argument('--force-setupname', type=str, default=None)
    parser.add_argument('--nchoose', type=int, default=nchoose)
    args = parser.parse_args()
    nchoose = args.nchoose
    assert len(args.time) == 5, "global_time should be 5 elements"
    convert(args.matfile, args.time, args.force_setupname)
