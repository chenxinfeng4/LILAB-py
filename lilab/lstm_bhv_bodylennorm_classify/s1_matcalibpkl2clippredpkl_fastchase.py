import pickle
import numpy as np
from lilab.lstm_bhv_bodylennorm_classify.utilities_package_feature import package_feature
import argparse
from lilab.lstm_bhv_bodylennorm_classify.s1_matcalibpkl2clippredpkl import (
    load, create_clips, load_engine, predict_cluster_labels
)


def main(project_dir):
    matcalibpkl_l, vnake_l, paramdict = load(project_dir)
    df_records, matdict = create_clips(matcalibpkl_l, vnake_l, paramdict)

    pkldata = pickle.load(open('/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/24FP-behCluster/2407FP-total/lstm_offline.clippredpkl', 'rb'))
    feat_clips = pkldata['feat_clips']
    feat_clips[:,:,:2] = np.clip(feat_clips[:,:,:2] * 1.5, 0, 0.16)
        
    trt_model = load_engine()
    predict_cluster_labels(project_dir, trt_model, df_records, feat_clips)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_dir', type=str)
    args = parser.parse_args()
    main(args.project_dir)
