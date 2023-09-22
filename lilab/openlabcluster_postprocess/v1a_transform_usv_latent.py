#python -m lilab.openlabcluster_postprocess.v1a_transform_usv_latent /A/B/C/
# %%
import pickle
import os.path as osp
import glob
import argparse


usv_pkl_folder = '/mnt/liying.cibr.ac.cn_Xiong/USV_MP4-toXiongweiGroup/Shank3_USV/usv_label/Shank3/'

def main(usv_pkl_folder):
    usv_pkl_files = glob.glob(osp.join(usv_pkl_folder, '*.pkl'))
    assert len(usv_pkl_files), 'No usv file found.'
    out_dict = {'video_nakes': [osp.basename(osp.splitext(f)[0]) for f in usv_pkl_files],
                'usv_latent': {
                    osp.basename(osp.splitext(f)[0]): pickle.load(open(f, 'rb')) for f in usv_pkl_files}}

    out_file = osp.join(usv_pkl_folder, 'usv_latent.usvpkl')
    pickle.dump(out_dict, open(out_file, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('usv_pkl_folder', type=str,
                        help='Folder containing pkl files of usv latent')
    args = parser.parse_args()
    main(args.usv_pkl_folder)
