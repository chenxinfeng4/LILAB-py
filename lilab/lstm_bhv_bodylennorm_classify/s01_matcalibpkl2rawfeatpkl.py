# conda activate OpenLabCluster
# python -m lilab.lstm_bhv_bodylennorm_classify.s01_matcalibpkl2rawfeatpkl $PROJECT_DIR
# project_dir="/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/24FP-behCluster/2407FP-total"

import pickle
import numpy as np
import os.path as osp
import glob
import tqdm
from lilab.lstm_bhv_bodylennorm_classify.utilities_package_feature import package_feature
import argparse
import multiprocessing


def load(project_dir):
    bodylength_pkl = osp.join(project_dir, 'bodylength.pkl')
    bodylength_data = pickle.load(open(bodylength_pkl, 'rb'))

    bodylength_l = bodylength_data['bodylength_l']
    sniff_zoom_l = bodylength_data['sniff_zoom_l']
    matcalibpkl_nake_l = bodylength_data['matcalibpkl_nake_l']

    matcalibpkl_l = glob.glob(osp.join(project_dir+'/*.smooth*.matcalibpkl'))
    vnake_l = [osp.basename(osp.splitext(f)[0]).split('.')[0] for f in matcalibpkl_l]
    assert set(vnake_l) <= set(matcalibpkl_nake_l)

    paramdict = {f:(bodylength_l[i], sniff_zoom_l[i]) for i, f in enumerate(matcalibpkl_nake_l)}
    return matcalibpkl_l, vnake_l, paramdict


def worker(args):
    matcalibpkl_nake, matcalibpkl, paramdict = args
    feat_l = dict()
    bodylength, sniff_zoom = paramdict[matcalibpkl_nake]
    p3d_CTK3 = pickle.load(open(matcalibpkl, 'rb'))['keypoints_xyz_ba'].transpose([1,0,2,3]).astype(float)
    feat_black_first = package_feature(p3d_CTK3, bodylength, sniff_zoom)
    feat_white_first = package_feature(p3d_CTK3[::-1], bodylength, sniff_zoom)
    feat_l[matcalibpkl_nake + '_blackFirst'] = feat_black_first.T.astype(np.float32)  #Tx32
    feat_l[matcalibpkl_nake + '_whiteFirst'] = feat_white_first.T.astype(np.float32)
    return feat_l


def inplace_calculate_feat_clips(df_records, feat_dict):
    df_records['feat_clip_index'] = np.arange(len(df_records))
    feat_clips = np.zeros([df_records.shape[0], 24, next(iter(feat_dict.values())).shape[1]], dtype=np.float32)
    for i, record in enumerate(tqdm.tqdm(df_records.itertuples(), total=len(df_records))):
        feat_wave = feat_dict[record.vnake + ('_blackFirst' if record.isBlack else '_whiteFirst')]
        feat_clip = feat_wave[record.startFrame:record.startFrame+24]
        feat_clips[i] = feat_clip
    
    return feat_clips


def main(project_dir):
    matcalibpkl_l, vnake_l, paramdict = load(project_dir)

    args_l = [(v, m, paramdict) for v, m in zip(vnake_l, matcalibpkl_l)]
    with multiprocessing.Pool(processes=40) as pool:
        results = list(tqdm.tqdm(pool.imap_unordered(worker, args_l), total=len(args_l)))
    feat_dict = dict()
    for result in results:
        feat_dict.update(result)
    
    feat_pkl = osp.join(project_dir, 'rawfeat.pkl')
    pickle.dump(feat_dict, open(feat_pkl, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_dir', type=str)
    args = parser.parse_args()
    main(args.project_dir)