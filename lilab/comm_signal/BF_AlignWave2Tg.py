import numpy as np
import matplotlib.pyplot as plt

def BF_AlignWave2Tg(Wave, Trigger, wds_L, wds_R, fs=1, ifshow=False):
    # check & adjust paras
    assert np.ndim(Wave) == 1
    assert np.ndim(Trigger) == 1
    assert np.isscalar(wds_L)
    assert np.isscalar(wds_R)
    assert wds_R > wds_L

    Trigger = np.floor(Trigger * fs).astype(int)
    wds_L = np.floor(wds_L * fs).astype(int)
    wds_R = np.floor(wds_R * fs).astype(int)

    if not ifshow:
        ifshow = False

    # clean out "Trigger" invalid head and tail
    len_ = len(Wave)
    ind_dirty = (Trigger + wds_L <= 0) | (Trigger + wds_R > len_)
    if np.any(ind_dirty):
        print(f'flag 超出窗口/总合 = {np.sum(ind_dirty):.0f}/{len(Trigger):.0f}, 自动清理')
        Trigger = Trigger[~ind_dirty]

    ind_dirty = np.isnan(Wave[Trigger + wds_L]) | np.isnan(Wave[Trigger + wds_R])
    if np.any(ind_dirty):
        print(f'有nan数据/总合 = {np.sum(ind_dirty):.0f}/{len(Trigger):.0f}，自动清理')
        Trigger = Trigger[~ind_dirty]

    ntrial = len(Trigger)
    assert ntrial >= 1, 'flag 数量太少'

    # calculate
    Alg_sXtrail = np.zeros((int(wds_R - wds_L), ntrial))
    for i in range(ntrial):
        ind_choose = Trigger[i] + np.arange(wds_L, wds_R)
        Alg_sXtrail[:, i] = Wave[ind_choose]

    # ifshow
    if ifshow:
        Alg_mean = np.mean(Alg_sXtrail, axis=1)

        plt.figure()
        plt.subplot(3, 1, 1)
        xtick = np.arange(wds_L, wds_R + 1)
        ytick = np.arange(1, ntrial + 1)
        plt.imshow(Alg_sXtrail.T, aspect='auto', extent=[xtick[0], xtick[-1], ytick[0], ytick[-1]])
        plt.xlabel('sample tick')
        plt.ylabel('trial (#)')
        plt.axis('tight')

        plt.subplot(3, 1, 2)
        plt.plot(xtick, Alg_sXtrail.T, color=0.2 * np.array([1, 1, 1]))
        plt.plot(xtick, Alg_mean, color=[1, 0, 0], linewidth=2)
        plt.xlabel('sample tick')
        plt.ylabel('Amp')

        plt.subplot(3, 1, 3)
        if ntrial <= 1:
            return

        Alg_sem = np.std(Alg_sXtrail, axis=1) / np.sqrt(ntrial)
        h_mean = plt.plot(xtick, Alg_mean)
        plt.errorbar(xtick, Alg_mean, yerr=Alg_sem, color=[1, 0, 0], linewidth=2)
        plt.ylabel('Amp')

    return Alg_sXtrail