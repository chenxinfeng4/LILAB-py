# python -m lilab.usv.a2_usv_tsv2pkl /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/USV
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import pickle
import os.path as osp
import os
import argparse
import tqdm

# %%
tsvfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/USV/T2023-01-01_16-12-14.tsv'

def converter(tsvfile):
    tsvdata = pd.read_csv(tsvfile, sep="\t")
    dur_thr = 0.01 #10ms
    datafilt = tsvdata[tsvdata['duration']>dur_thr][['duration', 'start time']]
    bin_time = 1
    finebin = 0.2 #0.2sec
    edges = np.arange(0, 900+1, finebin) # 15min

    cuts = pd.cut(datafilt['start time'], bins=edges, right=False, labels=False)
    dataevent = datafilt.groupby(cuts)
    dataeventcount = dataevent['duration'].count().astype(float)
    dataeventsum = dataevent['duration'].sum()
    dataeventcount = dataeventcount.reindex(edges, fill_value=0)
    dataeventsum = dataeventsum.reindex(edges, fill_value=0)


    # %% smooth
    dataeventcount_smooth = pd.Series(gaussian_filter1d(dataeventcount, 5), index=dataeventcount.index)
    dataeventsum_smooth = pd.Series(gaussian_filter1d(dataeventsum, 5), index=dataeventsum.index)
    edges_rough = np.arange(0, 900+1,1)

    bin_durations, bin_edges, binnumber = stats.binned_statistic(dataeventsum_smooth.index, dataeventsum_smooth.to_numpy(), statistic='sum', bins=edges_rough)
    bin_counts, _, _ = stats.binned_statistic(dataeventcount_smooth.index, dataeventcount_smooth.to_numpy(), statistic='sum', bins=edges_rough)
    index = bin_edges[:-1]

    # %% plot
    fontsize = 24
    plt.figure(figsize=(20, 10))
    plt.plot(edges, dataeventcount,  color=[0.5,0.5,0.5], label="original")
    plt.plot(edges_rough[:-1], bin_counts, color='r', label="smoothed")
    plt.xlabel('Time (sec)', fontsize=fontsize)
    plt.ylabel('Counts', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.title('Counts = {}'.format(np.floor(bin_counts.sum())), fontsize=fontsize)
    plt.savefig(osp.splitext(tsvfile)[0] + ".usv.jpg")
    plt.close()

    # %% save pandas
    dfout = pd.DataFrame({'start_time':index, 
                        'usv_count':bin_counts,
                        'usv_duration':bin_durations})

    sns.pointplot(x='start_time', y='usv_count', data=dfout)
    outpkl = osp.splitext(tsvfile)[0] + '.usvpkl'
    with open(outpkl, 'wb') as f:
        pickle.dump(dfout, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_or_folder', type=str)
    args = parser.parse_args()
    file_or_folder = args.file_or_folder
    assert os.path.exists(file_or_folder)
    if osp.isdir(file_or_folder):
        import glob
        files = glob.glob(osp.join(file_or_folder, '*.tsv'))
        assert files, 'no tsv files in folder {}'.format(file_or_folder)
    else:
        assert file_or_folder.endswith('.tsv'), 'invalid tsv file {}'.format(file_or_folder)
        files = [file_or_folder]
    
    for tsvfile in tqdm.tqdm(files):
        converter(tsvfile)
