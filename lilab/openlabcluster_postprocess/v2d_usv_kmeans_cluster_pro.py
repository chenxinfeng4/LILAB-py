import os
import os.path as osp
#from openlabcluster.utils import auxiliaryfunctions
#from auxiliaryfunctions import *
from os.path import join
import numpy as np
import sys
import random
from Bio.Cluster import kcluster
from preprocess_and_features_calculate_samedirect import  cluster_plot2
from auxiliaryfunctions import read_config
import umap
import pickle as pkl
import argparse
sys.path.append('/DATA/taoxianming/rat/script/openlabcluster/pc_feats41_svm-AgeDayAll_DecoderSeq')


k1=10
secondK=20

def main(dataPath):
    resPath=osp.join(dataPath, 'KMeans')
    os.makedirs(resPath,exist_ok=True)
    ##load data
    data=pkl.load(open(join(dataPath,'usv_latent.usvpkl'),'rb')) # ['video_nakes', 'usv_latent']
    ##
    feat_all=[]
    for vid in data['video_nakes']:
        feat_all.append(data['usv_latent'][vid])

    ##kmeans and umap
    data2=np.concatenate(feat_all,axis=0)
    ##
    reducer = umap.UMAP(random_state=1024)
    embedding = reducer.fit_transform(data2)
    print(embedding.shape)
    np.savez(join(resPath,'USV_latFeatAll-umap'),feat=data2,embedding=embedding)

    #####from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
    ##cluster 20 first for balance  #937*20=18740,k2=45,svm acc=0.927
    random.seed(1000) 
    lb, _, _ = kcluster(data2, k1, dist='e',npass=10)
    labFile=join(resPath,'kmeans_K1-'+str(k1)+'_labels')
    np.savez(labFile,labels=lb)
    pathFig=labFile+'.jpg'
    cluster_plot2(embedding,lb,pathFig)

    ##balance choosing
    from collections import Counter
    stat=Counter(lb) ##
    chooseNum=min(stat.values()) ##1041
    #chooseNum=np.median(list(stat.values()))
    print('each clusetr clips %s in %s clusters'%(chooseNum,k1))
    ##!!get choose indexes of each behavior label
    labelNums=np.unique(lb)
    chooseCluInds={};chooseClips=[]
    ##
    inds_chosen=[np.random.choice(a=np.argwhere(lb==li)[:,0], size=chooseNum, replace=False) for li in labelNums]
    ##cluster 100 on chosen data
    inds_chosen2=np.concatenate(inds_chosen,axis=0)
    data2_k100=data2[inds_chosen2]
    embedding_k100=embedding[inds_chosen2]
    data2_k100.shape
    
    ##cluster by second K
    random.seed(100)
    ##in encoder feature (hidden state) space
    lb_k2, error, nfound = kcluster(data2_k100, secondK, dist='e',npass=10)
    ##save labels and UMAP
    #labFile=join(resPath,'kmeansK2use-'+str(secondK)+'_fromK1-%s_K100_labels'%(k1))
    labFile=join(resPath,'kmeansK2use-'+str(secondK)+'_fromK1-%s_labels'%(k1))
    np.savez(labFile,labels=lb_k2,inds_choose=inds_chosen2)
    ####plot umap features with clustered labels color
    pathFig=labFile+'.jpg'
    cluster_plot2(embedding_k100,lb_k2,pathFig)

    ##SVM classification on all embedding features
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    ##
    def svc_predict2(data2_k100,lb_k2,data2):
        random.seed(1111)
        train_ratio=0.9
        sample_num=data2_k100.shape[0]
        inds=[i for i in range(sample_num)]
        inds_shuffle=np.random.choice(a=inds, size=sample_num, replace=False)
        train_inds=inds_shuffle[:int(sample_num*train_ratio)]
        test_inds=inds_shuffle[int(sample_num*train_ratio):]
        ##
        train_X=data2_k100[train_inds];train_y=lb_k2[train_inds]
        test_X=data2_k100[test_inds];test_y=lb_k2[test_inds]
        ##
        model = SVC()
        model.fit(train_X,train_y)
        predictions = model.predict(test_X)
        acc=accuracy_score(test_y,predictions)
        print(acc)
        ##
        model = SVC()
        model.fit(data2_k100,lb_k2)
        lb_svm=model.predict(data2)
        return([lb_svm,acc])

    ##SVM classifier on embedding
    lb_svm,acc=svc_predict2(data2_k100,lb_k2,data2)
    labFileF=join(resPath,'svm2allAcc%s_kmeansK2use-%s_fromK1-%s'%(round(acc,2),secondK,k1))
    np.savez(labFileF,labels=lb_svm)
    cluster_plot2(embedding,lb_svm,labFileF+'.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('usv_pkl_folder', type=str)
    args = parser.parse_args()
    assert osp.isdir(args.usv_pkl_folder)
    main(args.usv_pkl_folder)
