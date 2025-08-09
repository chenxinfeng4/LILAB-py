# from lilab.OpenLabCluster_train.a1_mirror_mutual_filt_clippredpkl import factory_label_mirror_start0
# from lilab.OpenLabCluster_train.a1_mirror_mutual_filt_clippredpkl import factory_label_mirror_equal_start0
#python -m lilab.OpenLabCluster_train.a1_mirror_mutual_filt_clippredpkl Representive_K36.clippredpkl --already-mirrored
import numpy as np
import pickle
import os
import os.path as osp
import pandas as pd
import argparse

from lilab.openlabcluster_postprocess.s1_merge_3_file import get_assert_1_file
from lilab.openlabcluster_postprocess.s1a_clipNames_inplace_parse import parse_name

# project_dir = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day55_Mix_analysis/SexAgeDay55andzzcWTinAUT_MMFF/result32/'
# clippredpkl = get_assert_1_file(osp.join(project_dir,'*.clippredpkl'))


def factory_label_mirror_start0(nK_mutual, nK_mirrorhalf, start=0):
    # start from 0
    nK = nK_mutual + nK_mirrorhalf * 2
    assert start in [0, 1], "start must be 0 or 1"
    ind_label_mirror = np.concatenate([np.arange(nK_mutual),
                                        np.arange(nK_mirrorhalf) + nK_mirrorhalf + nK_mutual,
                                        np.arange(nK_mirrorhalf) + nK_mutual])
    assert set(ind_label_mirror) == set(np.arange(nK))
    
    def fun_label_mirror(x_):
        x = np.array(x_) - start
        assert np.max(x) - start < nK
        y = np.zeros_like(x) - 1
        y[x >= 0] = ind_label_mirror[x[x >= 0]]
        y[y>=0] += start
        return  y
    return ind_label_mirror, fun_label_mirror


def factory_label_mirror_equal_start0(nK_mutual, nK_mirrorhalf, start=0):
    # start from 0
    nK = nK_mutual + nK_mirrorhalf * 2
    assert start in [0, 1], "start must be 0 or 1"
    def fun_label_equal(x):
        x = np.array(x)
        assert np.max(x) - start < nK
        y = x.copy()
        y[x >= start+nK_mutual+nK_mirrorhalf] -= nK_mirrorhalf
        y[x<start] = -1
        return y
    return fun_label_equal


def cal_df_repr(df_clipNames:pd.DataFrame, fun_label_mirror_start0) -> pd.DataFrame:
    assert {'vnake', 'startFrame', 'isBlack'}.issubset(df_clipNames.columns)
    df1 = df_clipNames.sort_values(by=['vnake', 'startFrame', 'isBlack']).reset_index()
    assert np.all(df1[['vnake', 'startFrame']].values[::2] == df1[['vnake', 'startFrame']].values[1::2])
    cluster_id_start0 = df1['cluster_id'].values - 1
    cluster_id_start0_reverse = cluster_id_start0.reshape(-1,2)[:,::-1].ravel()
    cluster_id_start0_mirror = fun_label_mirror_start0(cluster_id_start0)
    is_repr = cluster_id_start0_reverse==cluster_id_start0_mirror
    df1['is_repr'] = is_repr
    df1.sort_values(by='index', inplace=True)
    df1.index
    df_clipNames['is_repr'] = df1['is_repr'].values
    return df_clipNames


def filt_mirror_df(df_clipNames, nK_mutual, nK_mirror_half, start=1):
    _, fun_label_mirror = factory_label_mirror_start0(nK_mutual, nK_mirror_half)
    assert {'vnake', 'isBlack', 'startFrame', 'cluster_labels'} <= set(df_clipNames.columns)
    df_clipNames['_index'] = np.arange(len(df_clipNames))
    df_sort = df_clipNames.sort_values(by=['vnake', 'startFrame', 'isBlack'])
    assert np.all(df_sort[['vnake', 'startFrame']].values[::2] == df_sort[['vnake', 'startFrame']].values[1::2])
    new_label_sort = df_sort['cluster_labels'].values - start #start from 0
    new_label_W = new_label_sort[::2]
    new_label_B = new_label_sort[1::2]
    new_label_B_mirror = fun_label_mirror(new_label_B)
    ind_ismirror = new_label_W == new_label_B_mirror
    ind_ismirror2 = np.array([ind_ismirror, ind_ismirror]).T.ravel()
    df_sort['cluster_labels_clean'] = df_sort['cluster_labels'].copy()
    df_sort['cluster_labels_clean'].iloc[~ind_ismirror2] = -1
    repr_new_iter_perc = np.mean(ind_ismirror)
    df_sortback = df_sort.sort_values(by='_index')
    del df_sortback['_index']
    print(f'mean mirror of {len(df_sort)} samples', repr_new_iter_perc)
    return df_sortback



def repeat_ind(ind):
    return np.array([ind, ind]).T.ravel()


def get_nK_new(df_mapping):
    nK_mutual = df_mapping[df_mapping['isMutual']]['new_cluster_id'].max()
    nK_mirrorhalf = (df_mapping[df_mapping['isMirror']]['new_cluster_id'].max() - nK_mutual) // 2
    return nK_mutual, nK_mirrorhalf


def remap_oldlabel(ratA, ratB, df_mapping, start=0):
    nK_mutual, nK_mirrorhalf = get_nK_new(df_mapping)
    ratA_ = ratA - start + 1 # start from 1
    ratB_ = ratB - start + 1 # start from 1
    old_cluster_id = df_mapping['old_cluster_id'].values
    old_cluster_id_src1groups = df_mapping[df_mapping['isSplit']]['old_cluster_id'].values
    src_1_groups = old_cluster_id_src1groups % 100
    src_1_groups_set, src_1_groups_toN = np.unique(src_1_groups, return_counts=True)
    max_toN = src_1_groups_toN.max()
    src_1_groups_repeatdict = {k:np.array([k+1000*(i+1) for i in range(max_toN)]) for k in src_1_groups_set}
    ratA_expand = np.repeat(ratA_[:,None], max_toN, axis=1).astype(np.int32)
    ratB_expand = np.repeat(ratB_[:,None], max_toN, axis=1).astype(np.int32)
    ratA_expand[~np.isin(ratA_, src_1_groups_set), 1:] = -1
    ratB_expand[~np.isin(ratB_, src_1_groups_set), 1:] = -1
    for k in src_1_groups_set:
        ratA_expand[ratA_==k] = src_1_groups_repeatdict[k]
        ratB_expand[ratB_==k] = src_1_groups_repeatdict[k]

    ratA_expand[~np.isin(ratA_expand, np.unique(old_cluster_id))] = -1
    ratB_expand[~np.isin(ratB_expand, np.unique(old_cluster_id))] = -1

    map_dict = dict(zip(df_mapping['old_cluster_id'].values, df_mapping['new_cluster_id'].values))
    map_dict[-1] = -1
    ratA_expand_new = np.vectorize(map_dict.get)(ratA_expand) # start from 1
    ratB_expand_new = np.vectorize(map_dict.get)(ratB_expand) # start from 1
    _, fun_mirror = factory_label_mirror_start0(nK_mutual, nK_mirrorhalf, start=1)
    ratA_expand_new_mirror = fun_mirror(ratA_expand_new).astype(float)
    ratA_expand_new_mirror[ratA_expand_new_mirror<0] = np.nan
    bool_match = ratB_expand_new[:,None,:] == ratA_expand_new_mirror[:,:,None]
    isample, iA, iB = np.where(bool_match)
    assert(len(set(isample)) == len(isample))
    assert np.all(ratB_expand_new[isample, iB] == ratA_expand_new_mirror[isample, iA])
    ind_repr_clip = np.zeros(len(ratA), dtype=bool)
    ind_repr_clip[isample] = True
    ratA_new_repr = np.zeros(len(ratA), dtype=np.int32) - 1
    ratA_new_repr[isample] = ratA_expand_new[isample, iA]
    ratB_new_repr = np.zeros(len(ratA), dtype=np.int32) - 1
    ratB_new_repr[isample] = ratB_expand_new[isample, iB]
    ratA_old_repr = ratA_expand[:, 0].copy()
    ratA_old_repr[ind_repr_clip] = ratA_expand[isample, iA]
    ratB_old_repr = ratB_expand[:, 0].copy()
    ratB_old_repr[ind_repr_clip] = ratB_expand[isample, iB]

    assert ratA.min() == 1 and ratB.min() == 1
    return ind_repr_clip, ratA_new_repr, ratB_new_repr, ratA_old_repr, ratB_old_repr



def main2(clippredpkl):
    project_dir = osp.dirname(clippredpkl)
    clippreddata = pickle.load(open(clippredpkl, 'rb'))
    if 'df_clipNames' not in clippreddata:
        clippreddata['df_clipNames'] = parse_name(clippreddata['clipNames'])
    df_clipNames = clippreddata['df_clipNames']

    nK_mutual, nK_mirrorhalf = get_nK_new(df_mapping)
    df_sort = df_clipNames.sort_values(by=['vnake', 'startFrame', 'isBlack'])
    ind_sort = df_sort.index.values
    embedding = clippreddata['embedding'][ind_sort]
    embedding_d2 = clippreddata['embedding_d2'][ind_sort]
    cluster_labels = clippreddata['cluster_labels'][ind_sort]  #start from 1
    ratA = cluster_labels[::2]
    ratB = cluster_labels[1::2]
    df_mapping = clippreddata['df_mirror_id_mapping']

    ind_r, ratA_new_repr, ratB_new_repr, ratA_old_repr, ratB_old_repr = remap_oldlabel(ratA, ratB, df_mapping, start=1)
    ind_repr = repeat_ind(ind_r)
    cluster_label_new = np.array([ratA_new_repr, ratB_new_repr]).T.ravel()
    cluster_labels_part = cluster_label_new[ind_repr]
    df_sort_repr = df_sort[ind_repr]
    assert not np.any(cluster_labels_part==-1)

    df_cluster_name = df_mapping[['new_cluster_id', 'new_cluster_name']].sort_values(by='new_cluster_id').drop_duplicates()
    new_cluster_names = df_cluster_name[df_cluster_name['new_cluster_id']>0]['new_cluster_name'].values
    assert len(np.unique(new_cluster_names)) == len(new_cluster_names)

    new_cluster_names2 = []
    for new_cluster_name in new_cluster_names:
        old_ids = df_mapping[df_mapping['new_cluster_name']==new_cluster_name]['old_cluster_id'].values
        old_ids_str = ",".join([str(int(i)) for i in old_ids])
        new_cluster_names2.append(f'({old_ids_str}) {new_cluster_name}')

    #%% save data
    clippreddataNew = dict()
    cluster_labels_part_1start = cluster_labels_part - cluster_labels_part.min() + 1 #start from 1
    nK = cluster_labels_part_1start.max()
    clippreddataNew['ncluster'] = nK
    clippreddataNew['ntwin'] = clippreddata['ntwin']
    clippreddataNew['cluster_labels'] = cluster_labels_part_1start
    assert len(new_cluster_names) == clippreddataNew['ncluster']

    clippreddataNew['cluster_names'] = new_cluster_names2
    clippreddataNew['embedding'] = embedding[ind_repr]
    clippreddataNew['embedding_d2'] = embedding_d2[ind_repr]
    ind_rawclip = np.arange(len(clippreddata['clipNames']))[ind_sort][ind_repr]
    clippreddataNew['clipNames'] = clippreddata['clipNames'][ind_rawclip]
    clippreddataNew['ind_rawclip'] = ind_rawclip
    clippreddataNew['df_clipNames'] = df_sort_repr.reset_index(drop=True)
    clippreddataNew['nK_mutual'] = nK_mutual
    clippreddataNew['nK_mirror_half'] = nK_mirrorhalf

    prec = int(len(clippreddataNew['embedding']) / len(embedding) * 100)

    output_dir = osp.join(project_dir, f'representitive_k{nK}_filt_perc{prec}')
    os.makedirs(output_dir, exist_ok=True)
    clippredpklNew = osp.join(output_dir, f'Representive_K{nK}.clippredpkl')
    print(f'save to {clippredpklNew}')
    pickle.dump(clippreddataNew, open(clippredpklNew, 'wb'))


def main_already_mirrored(clippredpkl):
    project_dir = osp.dirname(clippredpkl)
    clippreddata = pickle.load(open(clippredpkl, 'rb'))
    if 'df_clipNames' not in clippreddata:
        clippreddata['df_clipNames'] = parse_name(clippreddata['clipNames'])
    df_clipNames = clippreddata['df_clipNames']

    nK_mutual = clippreddata['nK_mutual']
    nK_mirror_half = clippreddata.get('nK_mirror_half', clippreddata.get('nK_mirrorhalf',None))
    df_clipNames['cluster_labels'] = clippreddata['cluster_labels']
    df_clipNames_back = filt_mirror_df(df_clipNames, nK_mutual, nK_mirror_half, start=1)
    df_clipNames_back['_index'] = np.arange(len(df_clipNames_back))
    df_clipNames_sort = df_clipNames_back.sort_values(by=['vnake', 'startFrame', 'isBlack'])
    ind_rawclip = df_clipNames_sort[df_clipNames_sort['cluster_labels_clean']!=-1]['_index'].values

    #%% save data
    clippreddataNew = dict()
    cluster_labels_part = df_clipNames.loc[ind_rawclip]['cluster_labels'].values
    cluster_labels_part_1start = cluster_labels_part - cluster_labels_part.min() + 1 #start from 1
    nK = cluster_labels_part_1start.max()
    clippreddataNew['ncluster'] = nK
    clippreddataNew['ntwin'] = clippreddata['ntwin']
    clippreddataNew['cluster_labels'] = cluster_labels_part_1start
    clippreddataNew['cluster_names'] = clippreddata['cluster_names']
    if 'embedding' in clippreddata:
        clippreddataNew['embedding'] = clippreddata['embedding'][ind_rawclip]
    if 'embedding_d2' in clippreddata:
        clippreddataNew['embedding_d2'] = clippreddata['embedding_d2'][ind_rawclip]
    if 'feat_clips' in clippreddata:
        clippreddataNew['feat_clips'] = clippreddata['feat_clips'][ind_rawclip]
    if 'cluster_pvalues' in clippreddata:
        clippreddataNew['cluster_pvalues'] = clippreddata['cluster_pvalues'][ind_rawclip]
    clippreddataNew['clipNames'] = clippreddata['clipNames'][ind_rawclip]
    clippreddataNew['ind_rawclip'] = ind_rawclip
    clippreddataNew['df_clipNames'] = df_clipNames.loc[ind_rawclip].reset_index(drop=True)
    clippreddataNew['nK_mutual'] = nK_mutual
    clippreddataNew['nK_mirror_half'] = nK_mirror_half
    clippreddataNew['cluster_names_mutualmerge'] = clippreddata['cluster_names_mutualmerge']
    
    prec = int(len(clippreddataNew['cluster_labels']) / len(clippreddata['cluster_labels']) * 100)
    output_dir = osp.join(project_dir, f'representitive_k{nK}_filt_perc{prec}')
    os.makedirs(output_dir, exist_ok=True)
    clippredpklNew = osp.join(output_dir, f'Representive_K{nK}.clippredpkl')
    print(f'save to {clippredpklNew}')
    pickle.dump(clippreddataNew, open(clippredpklNew, 'wb'))


def main(clippredpkl):
    project_dir = osp.dirname(clippredpkl)
    clippreddata = pickle.load(open(clippredpkl, 'rb'))
    if 'df_clipNames' not in clippreddata:
        clippreddata['df_clipNames'] = parse_name(clippreddata['clipNames'])
    df_clipNames = clippreddata['df_clipNames']

    df_sort = df_clipNames.sort_values(by=['vnake', 'startFrame', 'isBlack'])
    ind_sort = df_sort.index.values
    embedding = clippreddata['embedding'][ind_sort]
    embedding_d2 = clippreddata['embedding_d2'][ind_sort]
    cluster_labels = clippreddata['cluster_labels'][ind_sort]  #start from 1

    #%% pick the representitives
    df_mapping = clippreddata['df_mirror_id_mapping']
    nK_mutual, nK_mirrorhalf = get_nK_new(df_mapping)
    # df_mapping = pd.read_csv('/home/liying_lab/chenxf/ml-project/论文图表/semisupervised分类/mirror_id_mapping.csv')
    map_dict = dict(zip(df_mapping['old_cluster_id'].values, df_mapping['new_cluster_id'].values))

    _, fun_label_mirror = factory_label_mirror_start0(nK_mutual, nK_mirrorhalf)


    new_label = np.vectorize(map_dict.get)(cluster_labels) - 1 #start from 0
    new_label[new_label<0]=-1
    new_label_W = new_label[::2]
    new_label_B = new_label[1::2]
    new_label_B_mirror = fun_label_mirror(new_label_B)

    ind_r = (new_label_W == new_label_B_mirror) & (new_label_W >= 0) & (new_label_B >= 0)
    ind_repr = repeat_ind(ind_r)

    cluster_labels_part = new_label[ind_repr] #start from 0
    df_sort_repr = df_sort[ind_repr]

    #%% rename new representives cluster
    df_cluster_name = df_mapping[['new_cluster_id', 'new_cluster_name']].sort_values(by='new_cluster_id').drop_duplicates()
    new_cluster_names = df_cluster_name[df_cluster_name['new_cluster_id']>0]['new_cluster_name'].values
    assert len(np.unique(new_cluster_names)) == len(new_cluster_names)

    new_cluster_names2 = []
    for new_cluster_name in new_cluster_names:
        old_ids = df_mapping[df_mapping['new_cluster_name']==new_cluster_name]['old_cluster_id'].values
        old_ids_str = ",".join([str(int(i)) for i in old_ids])
        new_cluster_names2.append(f'({old_ids_str}) {new_cluster_name}')

    #%% save data
    clippreddataNew = dict()
    cluster_labels_part_1start = cluster_labels_part - cluster_labels_part.min() + 1 #start from 1
    nK = cluster_labels_part_1start.max()
    clippreddataNew['ncluster'] = nK
    clippreddataNew['ntwin'] = clippreddata['ntwin']
    clippreddataNew['cluster_labels'] = cluster_labels_part_1start
    assert len(new_cluster_names) == clippreddataNew['ncluster']
    if 'feat_clips' in clippreddata:
        clippreddataNew['feat_clips'] = clippreddata['feat_clips'][ind_repr]
    if 'cluster_pvalues' in clippreddata:
        clippreddataNew['cluster_pvalues'] = clippreddata['cluster_pvalues'][ind_repr]
    clippreddataNew['cluster_names'] = new_cluster_names2
    clippreddataNew['embedding'] = embedding[ind_repr]
    clippreddataNew['embedding_d2'] = embedding_d2[ind_repr]
    ind_rawclip = np.arange(len(clippreddata['clipNames']))[ind_sort][ind_repr]
    clippreddataNew['clipNames'] = clippreddata['clipNames'][ind_rawclip]
    clippreddataNew['ind_rawclip'] = ind_rawclip
    clippreddataNew['df_clipNames'] = df_sort_repr.reset_index(drop=True)
    clippreddataNew['nK_mutual'] = nK_mutual
    clippreddataNew['nK_mirror_half'] = nK_mirrorhalf

    prec = int(len(clippreddataNew['embedding']) / len(embedding) * 100)

    output_dir = osp.join(project_dir, f'representitive_k{nK}_filt_perc{prec}')
    os.makedirs(output_dir, exist_ok=True)
    clippredpklNew = osp.join(output_dir, f'Representive_K{nK}.clippredpkl')
    print(f'save to {clippredpklNew}')
    pickle.dump(clippreddataNew, open(clippredpklNew, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("clippredpkl", type=str)
    parser.add_argument("--already-mirrored", action="store_true")
    args = parser.parse_args()
    assert osp.isfile(args.clippredpkl)
    if args.already_mirrored:
        main_already_mirrored(args.clippredpkl)
    else:
        main(args.clippredpkl)
