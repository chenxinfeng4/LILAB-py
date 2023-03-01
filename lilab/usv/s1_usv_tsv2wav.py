# python lilab.usv.s1_usv_tsv2wav.py usv.tsv
# %%
tsvfile = r"/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/USV/T2023-01-04_14-44-30.tsv"

# %%
# read tsv file by pandas
import pandas as pd
import numpy as np
from scipy.io.wavfile import write
from math import pi
import argparse
import os.path as osp


Fs=44100
duration_cut = 0.01 # >10ms
def read_tsv_file(tsvfile):
    df = pd.read_csv(tsvfile, sep='\t')
    assert 'duration' in df.columns
    assert 'start time' in df.columns
    classnames = df['class'].unique()
    classname_list = df['class'].tolist()
    dur_list = df['duration'].to_numpy()
    start_list = df['start time'].to_numpy()
    jitter_list = np.array([cls=='jitter' for cls in classname_list])
    dur_list = dur_list[jitter_list]
    start_list = start_list[jitter_list]
    return start_list, dur_list


def read_tsv_file_durationcut(tsvfile, dc=duration_cut):
    df = pd.read_csv(tsvfile, sep='\t')
    df2 = df[df['duration']>=dc][['duration','start time']] 
    dur_list = df2['duration'].to_numpy()
    start_list = df2['start time'].to_numpy()
    return start_list, dur_list


def generate_wav_file(start_list, dur_list, Fs=44100):
    t = np.arange(0, 0.2, 1/Fs)
    f = 1600
    amplitude = np.iinfo(np.int16).max
    x = amplitude *np.sin(2*pi*f*t)
    x = x.astype(np.int16)
    end_time = start_list[-1] + dur_list[-1] + 1
    x_zeros = np.zeros(int(end_time*Fs),dtype=np.int16)
    print('len(x_zeros):', len(x_zeros))
    for start, dur in zip(start_list, dur_list):
        x_zeros[int(start*Fs):int(start*Fs)+len(x)] = x
    return x_zeros


def main(tsvfile):
    start_list, dur_list = read_tsv_file(tsvfile)
    x = generate_wav_file(start_list, dur_list)
    outfile = osp.splitext(tsvfile)[0] + '_event.wav'
    write(outfile, Fs, x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tsvfile', type=str, help='tsv file')
    args = parser.parse_args()
    main(args.tsvfile)
