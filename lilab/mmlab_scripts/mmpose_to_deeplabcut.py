# %% imports
import os
import os.path as osp
import yaml
import glob
import pickle
import argparse
import numpy as np
import pandas as pd


def get_meta_file(joints_names, Scorer):
    meta = {'data': {'start': 1638087265.3643062,
            'stop': 1638088035.5226293,
            'run_duration': 770.1583230495453,
            'Scorer': Scorer,
            'DLC-model-config file': {'stride': 8.0,
                'weigh_part_predictions': False,
                'weigh_negatives': False,
                'fg_fraction': 0.25,
                'mean_pixel': [123.68, 116.779, 103.939],
                'shuffle': True,
                'snapshot_prefix': '/home/liying_lab/chenxinfeng/deeplabcut-project/abc-edf-2021-11-23/dlc-models/iteration-0/abcNov23-trainset95shuffle2/test/snapshot',
                'log_dir': 'log',
                'global_scale': 0.8,
                'location_refinement': True,
                'locref_stdev': 7.2801,
                'locref_loss_weight': 1.0,
                'locref_huber_loss': True,
                'optimizer': 'sgd',
                'intermediate_supervision': False,
                'intermediate_supervision_layer': 12,
                'regularize': False,
                'weight_decay': 0.0001,
                'crop_pad': 0,
                'scoremap_dir': 'test',
                'batch_size': 16,
                'dataset_type': 'imgaug',
                'deterministic': False,
                'mirror': False,
                'pairwise_huber_loss': True,
                'weigh_only_present_joints': False,
                'partaffinityfield_predict': False,
                'pairwise_predict': False,
                'all_joints': [[i] for i in range(len(joints_names))],
                'all_joints_names': joints_names,
                'dataset': 'training-datasets/iteration-0/UnaugmentedDataSet_abcNov23/abc_edf95shuffle2.mat',
                'init_weights': '/home/liying_lab/chenxinfeng/deeplabcut-project/abc-edf-2021-11-23/dlc-models/iteration-0/abcNov23-trainset95shuffle2/train/snapshot-350000',
                'net_type': 'resnet_50',
                'num_joints': len(joints_names),
                'num_outputs': 1},
                'fps': 10.0,
                'batch_size': 16,
                'frame_dimensions': (600, 800),
                'nframes': 46646,
                'iteration (active-learning)': 0,
                'training set fraction': 0.95,
                'cropping': False,
                'cropping_parameters': [0, 800, 0, 600]}}
    return meta


def search_nettype_iternum(yaml_file):
    # get the parent folder of yaml_file
    dlc_project_dir = osp.dirname(osp.abspath(yaml_file))
    pose_cfgfile = glob.glob(dlc_project_dir+'/dlc-models/iteration-0/*shuffle1/train/pose_cfg.yaml')
    assert len(pose_cfgfile)==1, "There should be only one pose_cfg.yaml file in the iteration-0 folder"

    # get the nettype from the pose_cfg.yaml
    with open(pose_cfgfile[0], 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        nettype = cfg["net_type"].replace("_", "")

    # get the iternumber
    snapshot_file = glob.glob(dlc_project_dir+'/dlc-models/iteration-0/*shuffle1/train/snapshot-*.meta')
    assert len(snapshot_file)>0, "There should be at least one snapshot file in the iteration-0 folder"
    iternum = [int(osp.basename(x).split('-')[1].split('.')[0]) for x in snapshot_file]
    iternum = max(iternum)
    return nettype, iternum


def main(yaml_file = 'config.yaml', pkl_file_or_folder = 'video.pkl'):
    # check the file or folder of pkl_file_or_folder
    if osp.isdir(pkl_file_or_folder):
        folder = pkl_file_or_folder
        pkl_files = glob.glob(folder+'/*.pkl')
        assert len(pkl_files)>0, "There should be at least one pkl file in the folder"
    elif osp.isfile(pkl_file_or_folder):
        pkl_files = [pkl_file_or_folder]
    else:
        raise ValueError("The pkl_file_or_folder should be a folder or a pkl file")
    
    # loop through the pkl files
    for pkl_file in pkl_files:
        convert(yaml_file, pkl_file)


def convert(yaml_file = 'config.yaml', pkl_file = 'video.pkl'):
    with open(yaml_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    bodyparts = cfg['bodyparts']

    # load the keypoints from the pkl file
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    assert len(data[0]) == len(bodyparts), "The number of bodyparts is not the same as the one in the template"
    nframe = len(data)

    # create new dataframe with multiple headers as 'scorer', 'bodyparts' and 'coords'
    nettype, iternum = search_nettype_iternum(yaml_file)
    scorer = f'DLC_{nettype}_{cfg["Task"]}{cfg["date"]}shuffle1_{iternum}'
    scores = [scorer]
    coords = ['x', 'y', 'likelihood']
    
    # create multiindex header from the above lists
    header = pd.MultiIndex.from_product([scores, bodyparts, coords], names=['scorer', 'bodyparts', 'coords'])
    df_out = pd.DataFrame(columns=header, dtype = 'float16', index=range(nframe))
    
    keypoints_all = np.array(data)  #nframe_by_nkeypoints_by_3
    keypoints_all[:,:,2] = np.clip(keypoints_all[:,:,2], 0, 1) #clip the likelihood to [0,1]
    keypoints_all[:,:,0:2] = np.round(keypoints_all[:,:,0:2], 1) #round the coordinates 
    keypoints_all[:,:,2] = np.round(keypoints_all[:,:,2], 2)     #round the likelihood
    for i, bodypart in enumerate(bodyparts):
        df_out.loc[:, (scorer, bodypart, 'x')] = keypoints_all[:, i, 0]
        df_out.loc[:, (scorer, bodypart, 'y')] = keypoints_all[:, i, 1]
        df_out.loc[:, (scorer, bodypart, 'likelihood')] = keypoints_all[:, i, 2]
    

    # save he dataframe to the csv file and h5 file
    h5_file_out = osp.splitext(pkl_file)[0] + scorer + '.h5'
    csv_file_out = osp.splitext(pkl_file)[0] + scorer + '.csv'
    df_out.to_hdf(h5_file_out, 'df_with_missing')
    df_out.to_csv(csv_file_out)

    # save the meta data to the yaml file
    meta = get_meta_file(joints_names=bodyparts, Scorer=scorer)
    meta_file_out = osp.splitext(pkl_file)[0] + scorer + '_meta.pickle'
    pd.to_pickle(meta, meta_file_out)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert the pkl file to the csv file')
    parser.add_argument('yaml_file', type=str, default='config.yaml', help='the yaml file')
    parser.add_argument('pkl_file', type=str, default='video.pkl', help='the pkl file or folder')
    args = parser.parse_args()
    main(args.yaml_file, args.pkl_file)