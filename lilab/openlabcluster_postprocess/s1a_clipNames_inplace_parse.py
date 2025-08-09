# python -m lilab.openlabcluster_postprocess.s1a_clipNames_inplace_parse /A/B/C.clippredpkl
import numpy as np
import pickle
import re
import pandas as pd
import argparse
import os.path as osp

clippredpkl_file = "/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/DayAll20230828/All-DecSeq/FWPCA0.00_P100_en3_hid30_epoch300-decSeqPC0.9_svm2allAcc0.96_kmeansK2use-43_fromK1-20_K100.clippredpkl"

#%%
def parse_name(clipNames) -> pd.DataFrame:
    pattern = re.compile(r'startFrameReal(\d+)')
    assert all('startFrame0' in clipName for clipName in clipNames), 'clipName must contain "startFrame0"'

    vnake_l = [osp.basename(c.split('.')[0]) for c in clipNames]
    isBlack_l = [('-blackFirst_' in c) for c in clipNames]
    startFrame_l = [int(re.search(pattern, c).group(1)) for c in clipNames]
    df_clipNames = pd.DataFrame({'vnake': vnake_l, 'isBlack': isBlack_l, 'startFrame': startFrame_l})
    df_sort = df_clipNames.sort_values(by=['vnake', 'startFrame', 'isBlack'])
    assert np.all(df_sort[['vnake','startFrame']].values[::2] == df_sort[['vnake','startFrame']].values[1::2])
    return df_clipNames


def main(clippredpkl_file):
    pkldata = pickle.load(open(clippredpkl_file, 'rb'))
    clipNames = pkldata['clipNames']
    pkldata['df_clipNames'] = parse_name(clipNames)
    pickle.dump(pkldata, open(clippredpkl_file, 'wb'))
    print('done')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('clippredpkl_file', type=str, help='clippredpkl file')
    args = parser.parse_args()
    main(args.clippredpkl_file)
