import numpy as np

def BF_AlignSg2Tg(Lick, Trgger, wds_L, wds_R, Fs=1):
    """
    对齐触发周围的事件。注意：没有检查左右窗口越界。

    :param Lick: 周围事件的时间戳 (1D 数组)
    :param Trgger: 触发事件的时间戳 (1D 数组)
    :param wds_L: 窗口左侧，如 [-2] (整数)
    :param wds_R: 窗口右侧，如 [2] (整数)

    :return Alg_cell: 周围时间沿触发的对齐 (1x cell of 1y nums)
    """
    assert np.ndim(Lick) == np.ndim(Trgger) == 1, "输入数据格式必须为1维向量"
    assert wds_L < wds_R, "窗口大小错位！"
    wds_L = wds_L * Fs
    wds_R = wds_R * Fs
    ntrial = len(Trgger)
    Alg_cell = [None] * ntrial
    for i in range(ntrial):
        tick_now = Trgger[i]
        dtick = Lick - tick_now
        ttick = dtick[(dtick > wds_L) & (dtick < wds_R)]
        Alg_cell[i] = ttick
    return Alg_cell
