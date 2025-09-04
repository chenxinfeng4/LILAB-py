#%%
import numpy as np
import pickle
import os.path as osp
import glob
import tqdm
import argparse


np_norm = lambda x: np.linalg.norm(x, axis=-1)

project = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day55_Mix_analysis/SexAgeDay55andzzcWTinAUT_MMFF/data'

def main(project, skip_frames):
    matcalibpkl_l = glob.glob(osp.join(project, '*smoothed_foot.matcalibpkl'))
    matcalibpkl_nake_l = [(osp.basename(matcalibpkl)).split('.')[0]
                        for matcalibpkl in matcalibpkl_l]
    assert len(matcalibpkl_l)>0
    bodylength_l = []
    sniff_zoom_l = []
    np.random.seed(10)
    body_perc_l = 95 #np.random.randint(90, 98, len(matcalibpkl_l))
    body_perc_l = [95] * len(matcalibpkl_l)
    for body_perc, matcalibpkl in zip(body_perc_l, tqdm.tqdm(matcalibpkl_l)):
        matcalibdata = pickle.load(open(matcalibpkl, 'rb'))
        p3d = matcalibdata['keypoints_xyz_ba']
        p3d = p3d[skip_frames:]
        bodylength_trace = np.linalg.norm(p3d[:,:,0,:] - p3d[:,:,5,:], axis=-1)
        bodylength = np.percentile(bodylength_trace.ravel(), body_perc)
        sniff_zoom_length = np.mean(np.median(np_norm(p3d[:,:,0,:2] - p3d[:,:,1,:2]), axis= -1))/2
        sniff_zoom_l.append(sniff_zoom_length)
        bodylength_l.append(bodylength)

    bodylength_l = np.array(bodylength_l)
    sniff_zoom_l = np.array(sniff_zoom_l)
    pickle.dump({'bodylength_l': bodylength_l, 
                'sniff_zoom_l': sniff_zoom_l,
                'matcalibpkl_nake_l': matcalibpkl_nake_l},
                open(osp.join(project, 'bodylength.pkl'), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project', type=str)
    parser.add_argument('--skip-frames', type=int, default=0)
    args = parser.parse_args()
    main(args.project, args.skip_frames)
