# python -m lilab.photometry.s2_photpkl_2_dFF A/B/C.photpkl
# %%
from scipy.signal import medfilt
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize
from scipy import ndimage
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import os.path as osp
from lilab.comm_signal.detectTTL import detectTTL
import argparse
from scipy.signal import butter, filtfilt
from scipy import ndimage

file_l = glob.glob('/mnt/liying.cibr.ac.cn_Data_Temp/LS_NAC_fiberphotometry/PHO_11_NAC/0324行为记录/*.photpkl')

# pklfile = file_l[0]


# %%
def plotHz(Fs, data_l, *args, **kwargs):
    plt.plot(np.arange(len(data_l))/Fs, data_l, *args, **kwargs)


#%%
def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
    '''Compute a double exponential function with constant offset.
    Parameters:
    t       : Time vector in seconds.
    const   : Amplitude of the constant offset. 
    amp_fast: Amplitude of the fast component.  
    amp_slow: Amplitude of the slow component.  
    tau_slow: Time constant of slow component in seconds.
    tau_multiplier: Time constant of fast component relative to slow. 
    '''
    tau_fast = tau_slow*tau_multiplier
    return const+amp_slow*np.exp(-t/tau_slow)+amp_fast*np.exp(-t/tau_fast)

class PhotFilter:
    """
    计算 Photometry 的 dF/F. 
        Step1：滤波
        Step2：计算&减去 bleach
        Step3：减去利用405运动噪声
        Step4：计算dF/F

    但是不考虑 offset
    
    参考代码：
        https://github.com/ThomasAkam/photometry_preprocessing/blob/master/Photometry%20data%20preprocessing.ipynb
    """
    color_l = ['k', 'g', 'C1']
    label_l = ['ref', 'sig_470','sig_565']

    def __init__(self, Fs, ref_405:np.ndarray, sigs:np.ndarray, show:bool=True):
        self.Fs = Fs
        self.ref_405 = ref_405
        self.timestamps = np.arange(len(ref_405))/Fs
        sigs = np.squeeze(sigs)
        assert sigs.ndim == 1 or sigs.ndim == 2
        if sigs.ndim == 1:
            sigs = sigs[None, :]
            self.sig_470 = sigs[0]
            self.sig_565 = sigs[0]*0
        else:
            self.sig_470 = sigs[0]
            self.sig_565 = sigs[1]

        self.sig_l = [self.ref_405, self.sig_470, self.sig_565]


        if show:
            self.show_raw()

    def p1_split(self):
        kernel=np.ones((60))
        kernel[30:] = -1
        kernel /= np.abs(kernel).sum()
        f_detect2 = ndimage.convolve(self.sig_l[2][:,None], kernel[:,None], mode='nearest')[:,0]
        f_detectabs = np.abs(f_detect2)
        thr = 1
        #thr = 0.8
        #thr = 0.5
        if f_detectabs.max() > thr:
            print('Big motion detected')
            up_thr = f_detectabs > thr
            tRise, tDur = detectTTL(up_thr)

            tcursor = 0
            subs_sig = []
            subs_timestamp = []
            sig_l = np.array(self.sig_l)
            for tRise_now, tDur_now in zip(tRise.astype(int), tDur.astype(int)):
                sig_l[:,tRise_now:tRise_now+tDur_now] = sig_l[:, [tRise_now]]
                sig_l_now = sig_l[:,tcursor:tRise_now+tDur_now]
                timestamp_now = self.timestamps[tcursor:tRise_now+tDur_now]
                subs_sig.append(sig_l_now)
                subs_timestamp.append(timestamp_now)
                tcursor = tRise_now+tDur_now
            
            subs_sig.append(sig_l[:,tcursor:])
            subs_timestamp.append(self.timestamps[tcursor:])

            subs_photFilter = []
            for sig_l, timestamps in zip(subs_sig, subs_timestamp):
                assert sig_l.shape[1] == len(timestamps)
                subPhotFilter = PhotFilter(self.Fs, sig_l[0], sig_l[1:], show=False)
                subPhotFilter.timestamps = timestamps
                subs_photFilter.append(subPhotFilter)
            
            return subs_photFilter

        else:
            return [self]

    @classmethod
    def p2_merge(cls, subs_photFilter:list):
        """
        Merge the photometric filters and the time series.
        """
        assert isinstance(subs_photFilter, list)
        assert all(isinstance(p, cls) for p in subs_photFilter)
        timestamp = np.concatenate([p.timestamps for p in subs_photFilter])
        assert len(np.unique(np.round(np.diff(timestamp), 3)))==1
        Fs = subs_photFilter[0].Fs
        sig_l = np.concatenate([np.array(p.sig_l) for p in subs_photFilter], axis=1)
        photFilter = PhotFilter(Fs, sig_l[0], sig_l[1:], show=False)
        photFilter.timestamps = timestamp
        photFilter.dfF_l = np.concatenate([np.array(p.dfF_l) for p in subs_photFilter], axis=1)
        photFilter.demotioned_l = np.concatenate([np.array(p.demotioned_l) for p in subs_photFilter], axis=1)
        return photFilter

    def show_raw(self):
        for ix in range(len(self.label_l)):
            plt.plot(self.timestamps, self.sig_l[ix], self.color_l[ix], label=self.label_l[ix])
        plt.legend()
        plt.ylabel('Signal (volts)')
        plt.xlabel('Time (sec)')
        plt.title('Raw trace')

    def s1_medfilt(self, show=False):
        self.denoise_l = [medfilt(sig, 5) for sig in self.sig_l]
        if show:
            for ix in range(len(self.label_l)):
                plt.plot(self.timestamps, self.denoise_l[ix], self.color_l[ix], label=self.label_l[ix])
            plt.legend()
            plt.xlabel('Time (sec)')
            plt.title('Step1: Median filtered')

    def s2_bleach(self, show=False):
        bleach_l = []
        detrended_l = []
        for sig in self.denoise_l:
            max_sig = np.max(sig)
            inital_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
            bounds = ([0      , 0      , 0      , 600  , 0],
                    [max_sig, max_sig, max_sig, 36000, 1])
            GCaMP_parms, parm_cov = curve_fit(double_exponential, self.timestamps, sig, 
                                            p0=inital_params, bounds=bounds, maxfev=4000)
            # GCaMP_parms, parm_cov = curve_fit(double_exponential, self.timestamps, sig, 
            #                                 p0=inital_params)#, bounds=bounds, maxfev=4000)
            GCaMP_expfit = double_exponential(self.timestamps, *GCaMP_parms)
            bleach_l.append(GCaMP_expfit)
            detrended_l.append(sig - GCaMP_expfit)
        self.bleach_l = bleach_l
        self.detrended_l = detrended_l

        if show:
            for ix in range(len(self.label_l)):
                plt.plot(self.timestamps, self.detrended_l[ix], self.color_l[ix], label=self.label_l[ix])
            plt.legend()
            plt.xlabel('Time (sec)')
            plt.title('Step2: Bleach filtered')

    def s3_motion(self, show=False):
        ref = self.detrended_l[0]
        sigs = self.detrended_l
        demotioned_l = sigs # copy, test
        demotioned_l = []
        for sig in sigs:
            slope, intercept, r_value, p_value, std_err = linregress(x=ref, y=sig)
            GCaMP_est_motion = intercept + slope * ref
            GCaMP_corrected = sig - GCaMP_est_motion
            demotioned_l.append(GCaMP_corrected)

        # demotioned_l = [medfilt(sig, 5) for sig in demotioned_l]
        # use low pass filter to remove the high frequency noise
        # low_pass=2
        # b, a = butter(2, low_pass/(0.5*self.Fs), 'low')
        # demotioned_l = [filtfilt(b,a,sig) for sig in demotioned_l]
        self.demotioned_l = demotioned_l
        self.s3_p_remove_outlier()

        if show:
            for ix in range(1, len(self.label_l)):
                plt.plot(self.timestamps, self.demotioned_l[ix], self.color_l[ix], label=self.label_l[ix])
            plt.legend()
            plt.xlabel('Time (sec)')
            plt.title('Step3: Motion correlated')

    def s3_p_remove_outlier(self, show=False):
        from scipy import signal
        thr = 2
        kernel = signal.triang(50)
        pad_width = 100
        kernel_padded = np.pad(kernel, (pad_width, pad_width), mode='constant')
        kernel_padded /= np.sum(kernel_padded)
        demotioned_l = self.demotioned_l
        filtered_l = []
        k1 = kernel_padded[:,None] / kernel_padded[:,None].sum()
        k2 = kernel_padded[::2][:,None] / kernel_padded[::2][:,None].sum()
        k4 = kernel_padded[::4][:,None] / kernel_padded[::4][:,None].sum()
        # k5 = kernel_padded[::8][:,None] / kernel_padded[::8][:,None].sum()
        for sig in demotioned_l:
            sig1 = ndimage.convolve(sig[:,None], k1, mode='nearest')[:,0]
            sig2 = ndimage.convolve(sig[:,None], k2, mode='nearest')[:,0]
            sig4 = ndimage.convolve(sig[:,None], k4, mode='nearest')[:,0]
            # sig5 = ndimage.convolve(sig[:,None], k5, mode='nearest')[:,0]

            sigmax = np.max(np.abs([sig1, sig2, sig4]), axis=0)
            print('sigmax', sigmax.max())
            index = sigmax > thr
            if np.any(index): 
                print('outlier detected')
                #dilate the index by size 10
                index_dialated = ndimage.binary_dilation(index, iterations=10)
                sig[index_dialated] = 0
                sig = medfilt(sig, 5)
            filtered_l.append(sig)
        self.demotioned_l = filtered_l

    def s4_dfF(self, show=False):
        self.dfF_l = [100*demotioned/bleach 
                        for demotioned, bleach in zip(self.demotioned_l, self.bleach_l)]
        if show:
            self.s4_dfF_show()
    
    def s4_dfF_show(self):
        for ix in range(1, len(self.label_l)):
            plt.plot(self.timestamps, self.dfF_l[ix], self.color_l[ix], label=self.label_l[ix])
            isNan = np.isnan(self.dfF_l[ix])
            if isNan.any():
                sig_isNan = np.zeros_like(isNan, dtype=float)
                sig_isNan[~isNan] = np.nan
                plt.plot(self.timestamps, sig_isNan+ix*0.2, '--',color=self.color_l[ix], label='LOSS')
        plt.legend()
        plt.ylabel('dF/F(%)')
        plt.xlabel('Time (sec)')
        plt.title('Step4: dF/F(%)')

    def s5_zscore(self, show=False):
        self.zscore_l = [(demotioned - np.mean(demotioned))/(np.std(demotioned) +0.0001)
                        for demotioned in self.demotioned_l]
        
        if show:
            for ix in range(1, len(self.label_l)):
                plt.plot(self.timestamps, self.zscore_l[ix], self.color_l[ix], label=self.label_l[ix])
            plt.legend()
            plt.ylabel('Signal (z-scored)')
            plt.xlabel('Time (sec)')
            plt.title('Step5: zscore')

    def s6_reset(self):
        self.dfF_l = [np.zeros_like(demotioned)+np.nan
                        for demotioned, bleach in zip(self.demotioned_l, self.bleach_l)]
        self.demotioned_l = self.dfF_l
    
# %%
def extract_dFF_ibrainarea(Fs, data, iarea, jpgfile):
    ref = data[iarea,0,0:int(Fs*15*60)]
    sigs = data[iarea,1:,0:int(Fs*15*60)]

    print(f'第{iarea}个脑区原始片段')
    # plt.figure(figsize=(10,10))
    # plt.subplot(3,2,1)
    rawphotFilter = PhotFilter(Fs, ref, sigs)
    # plt.xlabel(''); plt.xticks([])
    # plt.show()

    subs_photFilter = rawphotFilter.p1_split()

    for iseg, photFilter in enumerate(subs_photFilter):
        print(f'第{iarea}个脑区，第{iseg}/{len(subs_photFilter)}个片段')
        plt.figure(figsize=(10,10))
        plt.subplot(3,2,1)
        photFilter.show_raw()
        plt.xlabel(''); plt.xticks([])

        plt.subplot(3,2,2)
        photFilter.s1_medfilt(True)
        plt.xlabel(''); plt.xticks([])

        plt.subplot(3,2,3)
        photFilter.s2_bleach(True)
        plt.xlabel(''); plt.xticks([])

        plt.subplot(3,2,4)
        photFilter.s3_motion(True)
        plt.xlabel(''); plt.xticks([])

        plt.subplot(3,2,5)
        photFilter.s4_dfF(True)

        plt.subplot(3,2,6)
        photFilter.s5_zscore(True)
        # plt.show()
        plt.savefig(jpgfile)

    if len(subs_photFilter)>=2:
        # subs_photFilter[1].s6_reset()
        photFilter = PhotFilter.p2_merge(subs_photFilter)
        plt.figure(figsize=(8,2))
        photFilter.s4_dfF_show()
        # plt.show()
        plt.savefig(jpgfile)

    return np.array(photFilter.dfF_l)


def convert(pklfile):
    data_dict = pickle.load(open(pklfile, 'rb'))
    jpgfile = osp.splitext(pklfile)[0] + '_steps.jpg'
    Fs, data = data_dict['Fs'], data_dict['data']
    data[:,:,:5] = np.mean(data[:,:,5:15], axis=-1, keepdims=True)
    data[:,:,-15:] = np.mean(data[:,:,-35:-15], axis=-1, keepdims=True)
    dfF_data = []
    for iarea in range(len(data)):
        dfF_l = extract_dFF_ibrainarea(Fs, data, iarea, jpgfile)
        dfF_data.append(dfF_l)

    dfF_data = np.array(dfF_data)

    data_out = {
        'Fs': Fs,
        'data': dfF_data
    }

    narea = len(data)
    fig, axs = plt.subplots(3, 2, figsize=(10,8))
    for iarea, (rawF, dfF_l) in enumerate(zip(data, dfF_data)):
        plt.axes(axs[0, iarea])
        for ic, (color, label) in enumerate(zip(PhotFilter.color_l, PhotFilter.label_l)):
            plotHz(Fs, rawF[ic], color=color, label=label)
        plt.ylabel('Signal (Volt)')
        plt.legend()
        plt.title(f'The {iarea+1} brean area')
        # plt.xlim(300, 400)

        plt.axes(axs[1, iarea])
        for ic, (color, label) in enumerate(zip(PhotFilter.color_l, PhotFilter.label_l)):
            if ic==0: continue
            plotHz(Fs, dfF_l[ic], color=color, label=label)
            isNan = np.isnan(dfF_l[ic])
            if isNan.any():
                sig_isNan = np.zeros_like(isNan, dtype=float)
                sig_isNan[~isNan] = np.nan
                plotHz(Fs, sig_isNan+ic*0.2, '--',color=color, label='LOSS')
        # plt.xlim(360, 420)

        plt.axes(axs[2, iarea])
        for ic, (color, label) in enumerate(zip(PhotFilter.color_l, PhotFilter.label_l)):
            if ic==0: continue
            plotHz(Fs, dfF_l[ic], color=color, label=label)
            isNan = np.isnan(dfF_l[ic])
            if isNan.any():
                sig_isNan = np.zeros_like(isNan, dtype=float)
                sig_isNan[~isNan] = np.nan
                plotHz(Fs, sig_isNan+ic*0.2, '--',color=color, label='LOSS')
        
        plt.xlim([10, 110])
        plt.xlabel('Time (sec)')
        plt.ylabel('Signal (dF/F %)')
    plt.savefig(osp.splitext(pklfile)[0] + '.phot.jpg')

    outpklfile = osp.splitext(pklfile)[0] + '.dFFphotpkl'
    pickle.dump(data_out, open(outpklfile, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pklfile', help='pkl file to plot and convert to dF/F.')
    
    args = parser.parse_args()
    f_or_path = args.pklfile
    assert osp.exists(f_or_path), "photpkl_path not exists"
    if osp.isfile(f_or_path):
        file_l = [f_or_path]
    elif osp.isdir(f_or_path):
        file_l = glob.glob(osp.join(f_or_path, '*.photpkl'))
    else:
        raise ValueError("photpkl_path must be a file or a directory")
    
    for file in file_l:
        convert(file)
