# from lilab.comm_signal.detectTTL import detectTTL
import numpy as np

def detectTTL(data_1v:np.ndarray, adjacent_type:str=None, adjacent_value:float=0, fs=1):
    """过滤出TTL通道的数字信号，用上升沿和持续时间表述
    %[tRise_1y, tDur_1y] = detectTTL(data_1v)
    %[tRise_1y, tDur_1y] = detectTTL(data_1v, 'up-up', 20, fs=fs)
    %
    % ---------------------------Input---------------------------
    % data_1v     :  TTL通道的连续采样数据。或者 [bool 0 | bool 1]数据
    % adjacent    :  毗邻的类型['up-up', 'down-up', 'down-down']
    % adjacent_value: 毗邻的值 [>=0]
    %
    % ---------------------------Output--------------------------
    % tRise_1y    :  TTL的上升沿时刻，单位采样点
    % tDur_1y     :  TTL的高电平持续时长，单位采样点
    %
    """
    if not isinstance(data_1v, np.ndarray):
        data_1v = np.array(data_1v)
    assert data_1v.ndim == 1, 'data_1v must be 1-D array!'
    if data_1v.dtype == bool:
        TTL = data_1v
    else:
        TTL = data_1v > (data_1v.min() + data_1v.max())/2

    TTL = np.array([0, *TTL, 0])
    TTLdiff = np.diff(TTL)
    tRise_1y = np.where(TTLdiff == 1)[0]
    tDown_1y = np.where(TTLdiff == -1)[0]
    tDur_1y = tDown_1y - tRise_1y
    if fs==1:
        tRise_1y = tRise_1y.astype(int)
        tDown_1y = tDown_1y.astype(int)
    else:
        tRise_1y = tRise_1y/fs
        tDown_1y = tDown_1y/fs

    if len(tRise_1y)<=1 or adjacent_type is None:
        return tRise_1y, tDur_1y
    
    elif adjacent_type == 'up-up':
        tRest = tRise_1y[1:] - tRise_1y[:-1]
    elif adjacent_type == 'down-up':
        tRest = tRise_1y[1:] - tDown_1y[:-1]
    elif adjacent_type == 'down-down':
        tRest = tDown_1y[1:] - tDown_1y[:-1]
    else:
        raise ValueError('adjacent_type must be in [None, '
                         'up-up, down-up, down-down]')
    
    indmerge = tRest < adjacent_value
    indmerge_full = np.array([0, *indmerge, 0], dtype=bool)
    tRise_1y = tRise_1y[~indmerge_full[:-1]]
    tDown_1y = tDown_1y[~indmerge_full[1:]]
    tDur_1y = tDown_1y - tRise_1y

    return tRise_1y, tDur_1y
