import pickle
import numpy as np
import os.path as osp
import argparse

project_dir = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/24FP-behCluster/2407FP-total'


def main(project_dir, skip_frames, setspeed=None):
    rawfeat_pkl = osp.join(project_dir, 'rawfeat.pkl')

    rawfeat_data = pickle.load(open(rawfeat_pkl, 'rb'))
    rawfeat = np.concatenate([v[skip_frames:] for v in rawfeat_data.values()], axis=0)

    feat_speed = rawfeat[:, :2].ravel()
    feat_clip_speed_norm = np.percentile(feat_speed, 95)
    print('feat_clip_speed_norm', feat_clip_speed_norm)
    if setspeed is not None:
        feat_clip_speed_norm = setspeed
        print('feat_clip_speed_norm', feat_clip_speed_norm)
    for k, v in rawfeat_data.items():
        v[:, :2] = np.clip(v[:, :2] / feat_clip_speed_norm, 0, 4)
        print(k, v[:,:2].max())


    # save norm feat
    normfeat_pkl = osp.join(project_dir, 'rawfeat_norm.pkl')
    pickle.dump(rawfeat_data, open(normfeat_pkl, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project_dir', type=str, default=project_dir)
    parser.add_argument('--skip-frames', type=int, default=0)
    parser.add_argument('--setspeed', type=float, default=None)
    args = parser.parse_args()
    main(args.project_dir, args.skip_frames, args.setspeed)
