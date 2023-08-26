


'''
module load cuda/11.0.2  cudnn/8.1.0
source activate OpenLabCluster
cd /DATA/taoxianming/rat/script/openlabcluster/script
ipython
'''
#%%
import os
from os.path import join
import numpy as np
import cv2
from glob import glob as glb
import glob
import random
from shutil import copyfile
import sys
import shutil
from openlabcluster.utils import auxiliaryfunctions
import re

#for vf in vFolders:
#  videos=[i.split('/')[-1] for i in glob.glob(join(oriPath,vf,'*.mp4'))]
def write_video(videos,cp_clips,oriPath,writePath,fn):
  fps=30
  segLength=int(fps*0.8)
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  for vi,vid in enumerate(videos):
    print(vi)
    cap=cv2.VideoCapture(join(oriPath,vid))
    vName=re.sub('_1_400p.mp4','',vid)
    print(vName)
    ##!!!!!!!!get matched clips
    clipUsed=[c for c in cp_clips if c.find(vName)>-1 and c.find(fn)>-1]  ##'blackFirst' or 'whiteFirst'
    print(len(clipUsed))
    startFrames=[int(cu[cu.find('startFrameReal')+14:-4]) for cu in clipUsed]
    #clipUsedNames=[c+'.mp4' for c in clipUsed]
    clipUsedNames=clipUsed
    for i,sf in enumerate(startFrames):
      cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
      out = cv2.VideoWriter(join(writePath,clipUsedNames[i]), fourcc, fps,(400, 400))
      #out = cv2.VideoWriter(join(videoPath,clipUsedNames[i]), fourcc, fps,(320, 240))
      for _ in range(segLength):
        success, frame = cap.read()
        #frameR=frame
        #ddframeR=cv2.resize(frame, (600,600))
        out.write(frame)
      out.release()
    cap.release()

##
#project=sys.argv[1]
#project='/DATA/taoxianming/rat/result/openlabcluster/zhongzhenchao/20230319_LSlesion/feats35fs1.0s/LSlesion-2023-06-25'
#project='/DATA/taoxianming/rat/result/openlabcluster/chenxinfeng/sexualdimo_d35/feats35fs1.0s/sexualdimo_d35-2023-06-26'
#
#project='/DATA/taoxianming/rat/result/openlabcluster/chenxinfeng/sexualdimo_d35/feats35fs0.8s/sexualdimo_d35-2023-06-28'
#project='/DATA/taoxianming/rat/result/openlabcluster/zhongzhenchao/20230319_LSlesion/feats35fs0.8s/LSlesion-2023-06-28'
#project='/DATA/taoxianming/rat/result/openlabcluster/zhongzhenchao/LS-oxtr/all0627/feats35fs0.8s/LSoxtr-2023-06-28'
#project='/DATA/taoxianming/rat/result/openlabcluster/zhongzhenchao/LS-oxtr/all0629/feats35fs0.8s/LS-oxtr-2023-06-29'
project='/DATA/taoxianming/rat/result/openlabcluster/chenxinfeng/shank3Day36-2022-2023/feats35fs0.8s/Shank3Day36-2023-06-30'

#
#oriPath='/DATA/taoxianming/rat/data/chenxinfeng/sexualdimo_d35'
#oriPath='/DATA/taoxianming/rat/data/zhongzhenchao/20230319_LSlesion'
#oriPath='/DATA/taoxianming/rat/data/zhongzhenchao/LS-oxtr/all0627'
#oriPath='/DATA/taoxianming/rat/data/zhongzhenchao/LS-oxtr/all0627'
oriPath='/DATA/taoxianming/rat/data/chenxinfeng/shank3Day36-2022-2023'
#k_bests=[39]
##
pathFig=glob.glob(join(project,'output','kmeans','*getSecondK*_fromK100*'))[0].split('/')[-1]
k_bests=[int(pathFig[pathFig.find('SecondK')+7:pathFig.find('_fromK')])]

##
resPath=join(project,'output/kmeans')
os.makedirs(resPath,exist_ok=True)
config_yaml=join(project,'config.yaml')
##
cfg = auxiliaryfunctions.read_config(config_yaml)#load_config(config_yaml)
model_name = cfg['tr_modelName']
modelName=model_name.split('/')[-1]
clipNames=[l.split('/')[-1][:-1] for l in open(join(project,'videos/clipNames.txt'),'r').readlines()]

##
fsNames=['blackFirst','whiteFirst']
for fn in fsNames:
  for k_best in k_bests:
    clu_labs=np.load(join(resPath,modelName+'_kmeansK2use-'+str(k_best)+'HighDim_euc_bio_labels.npz'))['labels']+1
    videos=[vid.split('/')[-1] for vid in glob.glob(join(oriPath,'*.mp4'))]
    ##
    videoNum=40 #plot number
    for k in range(1,k_best+1):
      k_inds=list(np.argwhere(clu_labs==k)[:,0])
      if videoNum>len(k_inds):videoNum=len(k_inds)
      ##irreplace sampling
      choose_inds=random.sample(k_inds,videoNum)
      cp_clips=[clipNames[k_ind] for k_ind in choose_inds]
      # break
      vPath=join(resPath,'clu'+str(k))
      os.makedirs(vPath,exist_ok=True)
      write_video(videos,cp_clips,oriPath,vPath,fn)

    os.chdir(resPath)
    kmeanNpath=join(resPath,'kmeans_%s_%s'%(k_best,fn))
    os.makedirs(kmeanNpath,exist_ok=True)
    for k in range(1,k_best+1):
      kFile='clu%s.txt'%(k)
      fid=open(kFile,'w')
      vPath=join(resPath,'clu'+str(k))
      mp4Files=glb(join(vPath,'*.mp4'))
      for i,m in enumerate(mp4Files):
        #print(i)
        fid.write('file '+m+'\n')
      
      fid.close()
      ##merge clips tnto 1 for each cluster
      os.system('ffmpeg -f concat -safe 0 -i %s -c copy -f mp4 -y clu%s.mp4'%(kFile,k))
      os.system('ffmpeg -i clu%s.mp4  -y clu%s_400p.mp4'%(k,k))  #-s 400*400
      ##remove unused cluster clips folders
      shutil.rmtree(vPath)
      shutil.move('clu%s_400p.mp4'%(k),kmeanNpath)
      os.remove(kFile)
      os.remove('clu%s.mp4'%(k))



# %%
