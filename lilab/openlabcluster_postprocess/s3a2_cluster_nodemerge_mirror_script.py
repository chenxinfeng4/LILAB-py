# python -m lilab.openlabcluster_postprocess.s3a2_cluster_nodemerge_mirror *.clippredpkl
# %%
import pickle
import numpy as np
import os.path as osp
import argparse
import pandas as pd
from lilab.openlabcluster_postprocess.s1_merge_3_file import get_assert_1_file


clippredpklfile = '/mnt/liying.cibr.ac.cn_Xiong/USV_MP4-toXiongweiGroup/Shank3_USV/FWPCA0.00_P100_en3_hid30_epoch230_svm2allAcc0.93_kmeansK2use-44_fromK1-20_K100.clippredpkl'
merge_names_page = """
4	29	Behind or sniff tail (mirror)
5	43	Chase (mirror)
8	41	Pin stable (mirror)
10	22	Approach or face toward when a rat rearing (mirror)
17	40	Chase and sniff when a rat rearing (mirror)
19	25	Sniff or chase slowly (mirror)
26	42	Behind or sniffing tail when rearing (mirror)
27	34	Approach (mirror)
31	35	Both rearing with contact (mirror)
"""



# %%
def inplace_merge_nodes(clippredpklfile):
    project_dir = osp.dirname(clippredpklfile)
    seq_pkl = get_assert_1_file(osp.join(project_dir, '*_sequences.pkl'))
    merge_names = [s.strip() for s in merge_names_page.split("\n") if len(s)]
    new_label_name_list = []

    id_list = []
    for i in range(len(merge_names)):
        r = merge_names[i]
        *ids, new_label = r.split("\t")
        ids = [int(s) for s in ids]
        new_label_name_list.append(new_label)
        id_list.append(ids)
    id_list = np.array(id_list)

    clippreddata = pickle.load(open(clippredpklfile, "rb"))
    ncluster = clippreddata['ncluster']+1  #没有包含0

    id_sum = []
    id_list_ravel = id_list.ravel()
    for id_now in range(ncluster):
        if id_now not in id_list_ravel:
            id_sum.append([id_now])

    id_sum.extend(id_list.tolist())
    id_map = {}
    for newid, oldids in enumerate(id_sum):
        id_map.update({oldid:newid for oldid in oldids})
    id_map_np = np.array([[k,v] for k,v in id_map.items()])
    df_old_new_id_map = pd.DataFrame(id_map_np, columns=['old_id', 'new_id'])
    df_old_new_id_map = df_old_new_id_map.sort_values(by='old_id')
    df_old_new_id_map['cluster_names_old_start_0'] = [f'{s} [{d:2d}]' for d, s  in 
                                                    enumerate(['Far away non social'] + clippreddata['cluster_names'])]
    df_old_new_id_map['cluster_names_new_start_0'] = df_old_new_id_map['cluster_names_old_start_0'].copy()

    for ids, new_label_name in zip(id_list, new_label_name_list):
        df_old_new_id_map.loc[df_old_new_id_map['old_id'].isin(ids), 'cluster_names_new_start_0'] = new_label_name + f' [{ids[0]},{ids[1]}]'

    df_old_new_id_map.index = df_old_new_id_map['old_id'].values
    clippreddata['df_old_new_id_mirror_map'] = df_old_new_id_map
    pickle.dump(clippreddata, open(clippredpklfile, "wb"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('clippredpklfile', type=str)
    args = parser.parse_args()
    assert osp.isfile(args.clippredpklfile)
    inplace_merge_nodes(args.clippredpklfile)
