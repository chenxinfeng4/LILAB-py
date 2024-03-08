# python -m lilab.photometry.s1_tdms_2_photpkl A/B/C
# %%
from nptdms import TdmsFile
import glob
import os.path as osp
import pickle
import numpy as np
import argparse



def convert(tdms_f:str):
    tdms_file = TdmsFile.read(tdms_f, raw_timestamps=True)
    outfile = osp.splitext(tdms_f)[0]+'.photpkl'
    Fs = 10
    group = tdms_file['FberPhotometry']
    channels = group.channels()
    maxlen = len(group['Time']) // 3 * 3
    channel0_470 = group['Sig470-ROI0'][2:maxlen:3]
    channel0_405 = group['Sig405-ROI0'][1:maxlen:3]
    channel0_568 = group['Sig568-ROI0'][0:maxlen:3]

    channel1_470 = group['Sig470-ROI1'][2:maxlen:3]
    channel1_405 = group['Sig405-ROI1'][1:maxlen:3]
    channel1_568 = group['Sig568-ROI1'][0:maxlen:3]

    data = np.zeros((2, 3, maxlen//3))
    data[0, 0, :] = channel0_405
    data[0, 1, :] = channel0_470
    data[0, 2, :] = channel0_568
    data[1, 0, :] = channel1_405
    data[1, 1, :] = channel1_470
    data[1, 2, :] = channel1_568

    # check channel
    data_c = data.reshape(data.shape[0], -1)
    ind_c = np.mean(np.abs(data_c), axis=-1)>0.01
    print('Number of areas:', ind_c.sum())
    data = data[ind_c]
    datadict = dict()
    datadict['data'] = data
    datadict['Fs'] = Fs
    pickle.dump(datadict, open(outfile, 'wb'))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tdms_path', type=str)

    args = parser.parse_args()
    tdms_path = args.tdms_path
    assert osp.exists(tdms_path), "video_path not exists"
    if osp.isfile(tdms_path):
        tdms_path_l = [tdms_path]
    elif osp.isdir(tdms_path):
        tdms_path_l = glob.glob(osp.join(tdms_path, '*.tdms')) +\
                    glob.glob(osp.join(tdms_path, '*/*.tdms')) +\
                    glob.glob(osp.join(tdms_path, '*/*/*.tdms'))
    else:
        raise ValueError("tdms_path must be a file or a directory")
    
    for tdms_path in tdms_path_l:
        convert(tdms_path)
