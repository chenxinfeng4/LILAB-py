#!/usr/bin/python
'''
Author: your name
Date: 2021-11-13 19:17:10
LastEditTime: 2021-11-14 23:55:58
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \multiview_calib\examples\cxf_ball_rat\dlcBall_2_landmarks_ballglobal.py
'''
#python dlcBall_2_landmarks_ballglobal.py E:/cxf/dlc_ball_global

import json
import sys
import os
import glob
import pandas as pd
import numpy as np
views = ['c-1', 'c-2', 'c-3', 'c-4', 'c-5', 'c-6']

def view_link_csv(folder_csv):
    files_csv = glob.glob(os.path.join(folder_csv, '*.csv'))
    files_csv.sort()
    assert len(views)==len(files_csv)
    return dict(zip(views, files_csv))

def a1_load_dlc_csv(file):
    with open(file, 'r') as f:
        for iline, line in enumerate(f):
            if line.startswith('coords'):
                break
        else:
            raise 'Not Valid deeplabcut csv'
    df=pd.read_csv(file, header=list(range(iline)), skiprows=1)
    return df
    
def a2_flatten_points(df):
    arr = df
    for i in range(4):
        if 'x' in arr and 'y' in arr:
            break
        else:
            arr = arr.droplevel(level=0, axis=1)
    else:
        raise 'Not Valid deeplabcut csv'
    x = arr['x'].to_numpy()
    y = arr['y'].to_numpy()
    p = arr['likelihood'].to_numpy() if 'likelihood' in arr else (x*0+1.0)
    x_1v = x.flatten()
    y_1v = y.flatten()
    p_1v = p.flatten()
    return x_1v, y_1v, p_1v
    

def a3_thr_points(x_1v, y_1v, p_1v, thr=0.9):
    ids = np.arange(len(x_1v))
    idx_good = np.logical_not(np.isnan(p_1v))  & (p_1v > thr)
    px  = x_1v[idx_good]
    py  = y_1v[idx_good]
    ids = ids[idx_good]
    return px, py, ids
    
def a4_points_to_landmarksdict(px, py, ids):
    ids = ids.tolist()
    landmarks = np.array([px, py]).T.tolist()
    landmarksdict = {'ids':ids, 'landmarks':landmarks}
    return landmarksdict
    
def read_csv(file):
    df = a1_load_dlc_csv(file)
    x_1v, y_1v, p_1v = a2_flatten_points(df)
    px, py, ids = a3_thr_points(x_1v, y_1v, p_1v)
    landmarksdict = a4_points_to_landmarksdict(px, py, ids)
    return landmarksdict

def main(folder_csv, outjson):
    maps = view_link_csv(folder_csv)
    outdict = dict()
    for name, file in maps.items():
        print(name, os.path.split(file)[-1])
        outdict[name] = read_csv(file)

    with open(outjson, 'w') as f:
        json.dump(outdict, f, indent=4)
    
if __name__ == '__main__':
    n = len(sys.argv)
    if n == 1:
        folder = input("Choose the folder: >> ")
        if folder == None:
            exit()
        else:
            sys.argv.append(folder)
            
    print(sys.argv[1:])
    folder = sys.argv[1]
    assert sys.argv[2] in ['-o', '--out']
    outjson = sys.argv[3]
    main(folder, outjson)
    
