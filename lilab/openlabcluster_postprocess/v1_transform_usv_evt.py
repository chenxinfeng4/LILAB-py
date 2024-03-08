# python -m lilab.openlabcluster_postprocess.v1_transform_usv_evt /A/B/
# %%
import numpy as np
import pandas as pd
import glob
import os.path as osp
import matplotlib.pyplot as plt
import pickle
import argparse

usv_label_csv_folder = '/mnt/liying.cibr.ac.cn_Xiong/USV_MP4-toXiongweiGroup/Shank3_USV/usv_label/'

def main(usv_label_csv_folder):
    usv_files = glob.glob(osp.join(usv_label_csv_folder, '*.csv'))
    usv_files.sort(key=lambda x: osp.basename(osp.splitext(x)[0]))
    pd_files = [pd.read_csv(f) for f in usv_files]
    assert all({'start', 'end', 'main_freq', 'distribution'}<=set(df.columns) for df in pd_files)
    for usv_file, df in zip(usv_files, pd_files):
        df['video_nake'] = osp.basename(osp.splitext(usv_file)[0])

    df_all = pd.concat(pd_files)
    df_all['duration'] = df_all['end'] - df_all['start']
    df_all['idx_in_file'] = df_all['idx'].astype('int')
    del df_all['idx']
    
    # %%
    plt.scatter(df_all['duration'],df_all['main_freq'],  alpha=0.2)
    plt.hist2d(df_all['duration'], df_all['main_freq'], bins=(50, 50), cmap=plt.cm.jet, alpha=0.4)
    plt.xlabel('USV duration (s)')
    plt.ylabel('USV main frequency (kHz)')
    plt.savefig(osp.join(usv_label_csv_folder, 'usv_frequency_x_duration.jpg'))

    out_dict = {'video_nakes': list(np.array(df_all['video_nake'].unique())),
                'df_usv_evt': df_all}
    out_file = osp.join(usv_label_csv_folder, 'usv_evt.usvpkl')
    pickle.dump(out_dict, open(out_file, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('usv_label_csv_folder', type=str,
                        help='Folder containing csv files of usv labels')
    args = parser.parse_args()
    main(args.usv_label_csv_folder)
