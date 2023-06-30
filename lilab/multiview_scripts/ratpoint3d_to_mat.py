#!/usr/bin/python
#python -m lilab.multiview_scripts.ratpoint3d_to_mat rat_points3d_cm.json
import os
import sys
import json
import numpy as np
from scipy.io import savemat

npoints_per_rat = 14


def convert(jsonfile):
    with open(jsonfile, 'r') as f:
        data = json.load(f)

    ids = np.array(data['ids'])
    points_3d = np.array(data['points_3d'])
    nrat = np.ceil((np.max(ids)+1)/npoints_per_rat).astype('int')
    ids_full = np.arange(nrat * npoints_per_rat)
    points_3d_full = np.empty((nrat*npoints_per_rat, 3)) * np.nan
    points_3d_full[ids] = points_3d
    points_3d_nrat = points_3d_full.reshape(nrat, npoints_per_rat, 3)
    mdic = {'points_3d': points_3d_nrat}
    savemat(os.path.splitext(jsonfile)[0]+'.mat', mdic)
    

if __name__ == '__main__':
    n = len(sys.argv)
    if n == 1:
        folder = input("Select a json file: >> ")
        if folder == None:
            exit()
        else:
            sys.argv.append(folder)

    print(sys.argv[1:])

    jsonfile = os.path.abspath(sys.argv[1])
    convert(jsonfile)
