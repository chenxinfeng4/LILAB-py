
#conda activate DLC-GPU2.2-taoxm
#conda activate bcnet_bak_20211105
#conda activate base
#python
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from numpy import random
from scipy import ndimage
from sklearn import preprocessing,decomposition
import pywt
import seaborn as sns


def distance(p1,p2):
    #calculate distance between p1 and p2
    dist=np.dot(p1-p2,p1-p2)**0.5
    return(dist)

def distance_vect(vect):
    #calculate length of vect from zero point
    dist=np.dot(vect,vect)**0.5
    return(dist)

####preprocess####
def getHeadCenter(ratB):
    return(np.mean(ratB[:,[0,1,2]],axis=1))

#angle of 3-10 edge pairs
def getTrunkCenter(ratB):
    cents=np.mean(ratB[:,[3,4,5,6,8,10,12]],axis=1)
    return(cents)

def get_3partsCenter(ratB):
    ##head center, TrunkCenter_Neck, TrunkCenter_Tail
    points3=np.stack((np.mean(ratB[:,[0,1,2]],axis=1),
            np.mean(ratB[:,[3,6,8]],axis=1),
            np.mean(ratB[:,[5,10,12]],axis=1)),
            axis=1)
    return(points3)


def median_filter_5frames(ratB):
    ratB=ndimage.median_filter(ratB, size=(5,1,1)) ##filter by frames
    return(ratB)

    
##ego3D
def translation3D(ratB,cents):
    ##ratB:(frame,part_num,3)
    ##cents:(frame,3)
    ego3D=np.zeros(ratB.shape)#(frame,part_num,3)
    for i in range(ratB.shape[0]):
        ego3D[i]=ratB[i]-cents[i]
    return(ego3D)

def rotateXY_tail(ego3D):
    #tail:5
    ego3D_rot=np.zeros(ego3D.shape)#(frame,part_num,3)
    for i in range(ego3D.shape[0]):
        alpha=-math.atan(ego3D[i,5,1]/ego3D[i,5,0])
        #rotate clock-wise
        rot_mat=np.matrix([[math.cos(alpha),math.sin(alpha),0],[-math.sin(alpha),math.cos(alpha),0],[0,0,1]])
        ego3D_rot[i]=np.matmul(ego3D[i],rot_mat)
    return(ego3D_rot)


def nose2tail_length_median(ego3D):
    bsize=np.zeros((ego3D.shape[0],1))#(frame,part_num,3)
    for i in range(ego3D.shape[0]):
        bsize[i]=distance(ego3D[i,0],ego3D[i,5])##nose2tail distance
    return(np.median(bsize))


def rescale_nose2tail(ego3D):
    #nose:0,tail:5, nose2tail standard length 30
    stLen=30
    bsize_median=nose2tail_length_median(ego3D)
    scaleRatio=stLen/bsize_median
    ego3Drescaled=ego3D*scaleRatio
    return(ego3Drescaled)

def rotateXY_tail_sameDirect(ego3D):
    #tail:5
    ego3D_rot=np.zeros(ego3D.shape)#(frame,part_num,3)
    rot_alphas=np.zeros((ego3D.shape[0],1))
    for i in range(ego3D.shape[0]):
        ##rotate tail2center to y positive direction
        ##move to y-axis
        #alpha=angle_vectors2(np.array([0,1]),np.array([-ego3D[i,5,0],-ego3D[i,5,1]]))
        #if ego3D[i,5,0]>0: ##tail at region 2,3, counter clock-wise
        #    alpha=-alpha
        ##move to x-axis
        alpha=angle_vectors2(np.array([1,0]),np.array([-ego3D[i,5,0],-ego3D[i,5,1]]))
        if ego3D[i,5,1]<0: ##tail at region 1,2, counter clock-wise; 3,4 clock wise
            alpha=-alpha
        rot_alphas[i]=alpha*180/math.pi
        #rotate clock-wise
        rot_mat=np.matrix([[math.cos(alpha),math.sin(alpha),0],[-math.sin(alpha),math.cos(alpha),0],[0,0,1]])
        ego3D_rot[i]=np.matmul(ego3D[i],rot_mat)
    return([ego3D_rot,rot_alphas])


## 3D translate by trunkCenter
## horizontal rotation(x-y) rotate by trunkC2tailBase
## rescale by nose-tailbase length mean

def body_alignment2D_rescale(ratB,is_rescale=False):
    cents=getTrunkCenter(ratB)
    ego3D=translation3D(ratB,cents)
    ego3D_rot,rot_alphas=rotateXY_tail_sameDirect(ego3D)
    #body size rescale, by nose2tail length
    if is_rescale:
        ego3D_rot=rescale_nose2tail(ego3D_rot)
    return([ego3D_rot,rot_alphas])


##
def get_BodyVects(ratB):
	##spine vectors
    f_num=ratB.shape[0]
    vect_num=3
    dim=3
    vects=np.zeros((f_num,vect_num,dim)) #(frame,vect_num,3)
    for i in range(f_num):
        #neck-->nose
        vects[i,0]=ratB[i,0]-ratB[i,3]
        #body-->neck
        vects[i,1]=ratB[i,3]-ratB[i,4]
        #tail-->body
        vects[i,2]=ratB[i,4]-ratB[i,5]
    return(vects)


def get_BodyVects_project4_Angles(vects):
    ###get defined vect angles
    f_num=vects.shape[0]
    angle_num=2
    angles=np.zeros((f_num,angle_num,4))#(frame,angle_num=12)
    for i in range(f_num):
        ##spine
        #neck->nose <-> body->neck
        angles[i,0]=angle_project4_vectors(vects[i,0],vects[i,1])
        #body->neck <-> tail->body
        angles[i,1]=angle_project4_vectors(vects[i,1],vects[i,2])     
    return(angles)

####solo features####
##get defined vectors
def get_DefVects(ego3D):
    ###get defined vects
    f_num=ego3D.shape[0]
    vect_num=13
    dim=3
    vects=np.zeros((f_num,vect_num,dim)) #(frame,vect_num=13,3)
    for i in range(f_num):
        ##head vectors: 30 31 32
        nose=ego3D[i,0]
        vects[i,0]=nose-ego3D[i,3] #neck->nose----
        vects[i,1]=ego3D[i,1]-nose #nose->earL
        vects[i,2]=ego3D[i,2]-nose #nose->earR
        ##trunk vectors:43 46 48, 54 410 412
        body=ego3D[i,4]
        vects[i,3]=ego3D[i,3]-body #body->neck----
        vects[i,4]=ego3D[i,6]-body #body->shoulderL
        vects[i,5]=ego3D[i,8]-body #body->shoulderR
        vects[i,6]=body-ego3D[i,5] #tail->body----
        vects[i,7]=ego3D[i,10]-body #body->kneeL
        vects[i,8]=ego3D[i,12]-body #body->kneeR
        ##palm: front: 67 89
        vects[i,9]=ego3D[i,7]-ego3D[i,6] #shoulderL->palmL
        vects[i,10]=ego3D[i,9]-ego3D[i,8] #shoulderR->palmR
        ##foot: behind: 1011 1213
        vects[i,11]=ego3D[i,11]-ego3D[i,10] #kneeL->fooltL
        vects[i,12]=ego3D[i,13]-ego3D[i,12] #kneeR->footR
    return(vects)


def get_DefVectsLength(vects):
    ###get vect lengths
    ##vects:(frame,vect_num=13,3)
    f_num,vect_num=vects.shape[0:2]
    vects_lens=np.zeros((f_num,vect_num))#(frame,vect_num=13)
    for i in range(f_num):
        for j in range(vect_num):
            vects_lens[i,j]=distance_vect(vects[i,j])
    return(vects_lens)


'''
def get_DefVectsAngles(vects):
    ###get defined vect angles
    ##ratB:(frame,point_num,3)
    #vects=get_DefVects(ratB)
    ##vects:(frame,vect_num=13,3)
    #f_num=ratB.shape[0]
    f_num=vects.shape[0]
    angle_num=12
    angles=np.zeros((f_num,angle_num))#(frame,angle_num=12)
    for i in range(f_num):
        ##spine
        #neck->nose <-> body->neck
        angles[i,0]=angle_vectors(vects[i,0],vects[i,3])
        #body->neck <-> tail->body
        angles[i,1]=angle_vectors(vects[i,3],vects[i,6])        
        #
        ##head part
        #neck->nose <-> nose->earL
        angles[i,2]=angle_vectors(vects[i,0],vects[i,1])
        #neck->nose <-> nose->earR
        angles[i,3]=angle_vectors(vects[i,0],vects[i,2])
        #
        #
        ##up body
        #body->neck <-> body->shoulderL
        angles[i,4]=angle_vectors(vects[i,3],vects[i,4])
        #body->neck <-> body->shoulderR
        angles[i,5]=angle_vectors(vects[i,3],vects[i,5])
        ##down body
        #tail->body <-> body->kneeL
        angles[i,6]=angle_vectors(vects[i,6],vects[i,7])
        #tail->body <-> body->kneeR
        angles[i,7]=angle_vectors(vects[i,6],vects[i,8])
        #
        #
        ##palm
        #body->shoulderL <-> shoulderL->palmL
        angles[i,8]=angle_vectors(vects[i,4],vects[i,9])
        #body->shoulderR <-> shoulderR->palmR
        angles[i,9]=angle_vectors(vects[i,5],vects[i,10])
        ##foot
        #body->kneeL <-> kneeL->footL
        angles[i,10]=angle_vectors(vects[i,7],vects[i,11])
        #body->kneeR <-> kneeR->footR
        angles[i,11]=angle_vectors(vects[i,8],vects[i,12])
    return(angles)
'''


def get_DefVects_project4_Angles(vects):
    ###get defined vect angles
    '''
    ##ratB:(frame,point_num,3)
    #vects=get_DefVects(ratB)
    ##vects:(frame,vect_num=13,3)
    #f_num=ratB.shape[0]
    '''
    f_num=vects.shape[0]
    angle_num=12
    angles=np.zeros((f_num,angle_num,4))#(frame,angle_num,plane_num)
    for i in range(f_num):
        ##spine
        #neck->nose <-> body->neck
        angles[i,0]=angle_project4_vectors(vects[i,0],vects[i,3])
        #body->neck <-> tail->body
        angles[i,1]=angle_project4_vectors(vects[i,3],vects[i,6])        
        #
        ##head part
        #neck->nose <-> nose->earL
        angles[i,2]=angle_project4_vectors(vects[i,0],vects[i,1])
        #neck->nose <-> nose->earR
        angles[i,3]=angle_project4_vectors(vects[i,0],vects[i,2])
        #
        #
        ##up body
        #body->neck <-> body->shoulderL
        angles[i,4]=angle_project4_vectors(vects[i,3],vects[i,4])
        #body->neck <-> body->shoulderR
        angles[i,5]=angle_project4_vectors(vects[i,3],vects[i,5])
        ##down body
        #tail->body <-> body->kneeL
        angles[i,6]=angle_project4_vectors(vects[i,6],vects[i,7])
        #tail->body <-> body->kneeR
        angles[i,7]=angle_project4_vectors(vects[i,6],vects[i,8])
        #
        #
        ##palm
        #body->shoulderL <-> shoulderL->palmL
        angles[i,8]=angle_project4_vectors(vects[i,4],vects[i,9])
        #body->shoulderR <-> shoulderR->palmR
        angles[i,9]=angle_project4_vectors(vects[i,5],vects[i,10])
        ##foot
        #body->kneeL <-> kneeL->footL
        angles[i,10]=angle_project4_vectors(vects[i,7],vects[i,11])
        #body->kneeR <-> kneeR->footR
        angles[i,11]=angle_project4_vectors(vects[i,8],vects[i,12])
    return(angles)


def angle_project4_vectors(vector1,vector2):
    ang=angle_vectors(vector1,vector2)
    ## x-y plane angles
    ang_xy=angle_vectors(vector1[:2],vector2[:2])
    ## x-z plane
    ang_xz=angle_vectors(vector1[[0,2]],vector2[[0,2]])
    ## y-z plane
    ang_yz=angle_vectors(vector1[[1,2]],vector2[[1,2]])
    return(np.array([ang,ang_xy,ang_xz,ang_yz]))


def get_DefVects_project4_Angles_between2rats(ratB_defVects,ratW_defVects):
    f_num=ratB_defVects.shape[0]
    v_num_B=ratB_defVects.shape[1]
    v_num_W=ratW_defVects.shape[1]
    angles=np.zeros((f_num,v_num_B,v_num_W,4))#(frame,angle_num,plane_num)
    for fi in range(f_num):
        for bi in range(v_num_B):
            for wi in range(v_num_W):
                angles[fi,bi,wi]=angle_project4_vectors(ratB_defVects[fi,bi],ratW_defVects[fi,wi])
    return(angles)


def angle_vectors2(vector1,vector2):
    #calculate angle between vectors
    numerator=np.dot(vector1, vector2)
    denominat=np.dot(vector1,vector1)**0.5*np.dot(vector2,vector2)**0.5
    if denominat==0:denominat=0.00001
    cosTheta=round(numerator/denominat,10)
    cosTheta=1 if cosTheta>1 else cosTheta
    cosTheta=-1 if cosTheta<-1 else cosTheta
    return(math.acos(cosTheta)*180/math.pi)

##0-180
def angle_vectors(vector1,vector2):
    #calculate angle between vectors
    numerator=np.dot(vector1, vector2)
    denominat=np.dot(vector1,vector1)**0.5*np.dot(vector2,vector2)**0.5
    if denominat==0:denominat=0.00001
    cosTheta=round(numerator/denominat,10)
    cosTheta=-1 if cosTheta<-1 else cosTheta
    cosTheta=1 if cosTheta>1 else cosTheta
    return(math.acos(cosTheta)*180/math.pi)

'''
##0-360
def angle_vectors(v1, v2):
    numerator=np.dot(v1, v2)
    denominat=np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2)
    if denominat==0:denominat=0.00001
    cosTheta=round(numerator/denominat,10)
    if cosTheta>1:cosTheta=1
    if cosTheta<-1:cosTheta=-1
    ##
    deg = math.acos(cosTheta) * 180 / np.pi
    a1 = np.array([*v1, 0])
    a2 = np.array([*v2, 0])
    a3 = np.cross(a1, a2)
    if np.sign(a3[2]) < 0:deg = 360 - deg
    return deg
'''

def get_point_velocity(cents,fps): #one 3D point velicity
    #cents:(frame,dim=3)
    #cents_v:(frame,dim=3)
    time_gap=1/fps; #fps=15
    cents_v1=np.diff(cents,axis=0)/time_gap
    cents_v=np.concatenate([np.expand_dims(cents_v1[0,:],axis=0),cents_v1],axis=0)
    return(cents_v)


def get_speed_from_velocity(cents_v):
    speed=np.array([np.dot(a,a)**0.5 for a in cents_v])
    return(speed)


def get_speed_from_velocity_multipoints(cents_v):
    f,p=cents_v.shape[:2]
    speed=np.zeros((f,p))
    for fi in range(f):
        for pi in range(p):
            speed[fi,pi]=np.dot(cents_v[fi,pi],cents_v[fi,pi])**0.5
    return(speed)


import pandas as pd
def moving_mean_sd_frames(cents_v,frame_window):
    ##frame_window=3
    dim=cents_v.shape[1]
    mm=np.array([pd.Series(cents_v[:,j]).rolling(window=frame_window,center=True,axis=0, min_periods=1).mean() for j in range(dim)]).T
    ms=np.array([pd.Series(cents_v[:,j]).rolling(window=frame_window,center=True,axis=0, min_periods=1).var(ddof=0) for j in range(dim)]).T
    ##pandas std: default ddof=1, ddof=0 in numpy
    return([mm,ms])


def moving_mean_sd_frames3_2d(cents_v,frame_windows):
    ##cents_v: (frame_num,dim)
    dim=cents_v.shape[1]
    nfw=len(frame_windows)
    mms=np.zeros((cents_v.shape[0],dim,nfw,2))
    #(frame_num,dim,frame_windows_num,moving_mean_var)
    for fwi in range(nfw):
        mm=np.array([pd.Series(cents_v[:,j]).rolling(window=frame_windows[fwi],center=True,axis=0, min_periods=1).mean() for j in range(dim)]).T
        ms=np.array([pd.Series(cents_v[:,j]).rolling(window=frame_windows[fwi],center=True,axis=0, min_periods=1).std(ddof=0) for j in range(dim)]).T
        ##pandas std: default ddof=1, ddof=0 in numpy
        mms[:,:,fwi,0]=mm
        mms[:,:,fwi,1]=ms
    return(mms)


def moving_mean_sd_frames3_1d(cents_s,frame_windows):
    ##cents_v: (frame_num)
    nfw=len(frame_windows)
    mms=np.zeros((cents_s.shape[0],nfw,2))
    #(frame_num,frame_windows_num,moving_mean_var)
    for fwi in range(nfw):
        mm=np.array(pd.Series(cents_s).rolling(window=frame_windows[fwi],center=True,axis=0, min_periods=1).mean())
        ms=np.array(pd.Series(cents_s).rolling(window=frame_windows[fwi],center=True,axis=0, min_periods=1).std(ddof=0))
        ##pandas std: default ddof=1, ddof=0 in numpy
        mms[:,fwi,0]=mm
        mms[:,fwi,1]=ms
    return(mms)


####interaction features####
##0-13:nose,earL,earR,neck,body,tail,
#shoulderL,palmL,shoulderR,palmR,kneeL,footL,kneeR,footR
##interaction feature calculation
'''
def angle_points(line1_p1,line1_p2,line2_p1,line2_p2):
    #calculate angle between vectors calculated by points:p2->p1
    vector1=np.array([line1_p1[i]-line1_p2[i] for i in range(len(line1_p1))])#3d
    vector2=np.array([line2_p1[i]-line2_p2[i] for i in range(len(line2_p1))])
    numerator=np.dot(vector1, vector2)
    denominat=np.dot(vector1,vector1)**0.5*np.dot(vector2,vector2)**0.5
    cosTheta=round(numerator/denominat,10)
    return(math.acos(cosTheta)*180/math.pi)
'''


def angle_lineBypoints(line1_p1,line1_p2,line2_p1,line2_p2):
    #calculate angle between vectors calculated by points:p2->p1
    vector1=line1_p1-line1_p2#3d
    vector2=line2_p1-line2_p2
    return(angle_vectors(vector1,vector2))



#get distance of each two points between rats
def get_distances_BetweenPoints(ratB,ratW):
    frame_num,point_num=ratB.shape[0:2]
    ##
    points_dists=np.zeros((frame_num,point_num,point_num))
    for fi in range(frame_num):
        for pbi in range(point_num):
            for pwi in range(point_num):
                points_dists[fi,pbi,pwi]=distance(ratB[fi,pbi],ratW[fi,pwi])
    return(points_dists)
    


def get_distances_BetweenPoints2(ratB,ratW):
    frame_num,point_num=ratB.shape[0:2]
    point_num2=ratW.shape[1]
    ##
    points_dists=np.zeros((frame_num,point_num,point_num2))
    for fi in range(frame_num):
        for pbi in range(point_num):
            for pwi in range(point_num2):
                points_dists[fi,pbi,pwi]=distance(ratB[fi,pbi],ratW[fi,pwi])
    return(points_dists)

##calculate distances velicity between points (frame_num,14,14,4)

def velocity_distances_BetweenPoints(points_dists):
    frame_num,point_num=points_dists.shape[0:2]
    v_type=4
    points_dists_v=np.zeros((frame_num,v_type,point_num,point_num))
    ##frame 0123 or -1-2-3-4, no velocity, give -1
    for fi in range(frame_num):
        if fi<4 or fi>(frame_num-5): 
            continue
        points_dists_v[fi]=velocityByframes(points_dists[fi-4:fi+5])
    points_dists_v=startEndFill(points_dists_v)
    return(points_dists_v)


def get_vectsAnglesBetween2rats(ratB,ratW):
    vectsB=getVects(ratB)
    vectsW=getVects(ratW)
    frame_num,vect_num=vectsB.shape[0:2]
    vects_angles=np.zeros((frame_num,vect_num,vect_num))
    for fi in range(frame_num):
        for vbi in range(vect_num):
            for vwi in range(vect_num):
                vects_angles[fi,vbi,vwi]=angle_vectors(vectsB[fi,vbi],vectsW[fi,vwi])
    return(vects_angles)


def velocity_vectsAnglesBetween2rats(vects_angles):
    #vects_angles=get_vectsAnglesBetween2rats(ratB,ratW)
    frame_num,vect_num=vects_angles.shape[0:2]
    ##calculate angles velocity between spine vectors (frame_num,3,3,4)
    v_type=4
    vects_angles_v=np.zeros((frame_num,v_type,vect_num,vect_num))
    ##frame 0123 or -1-2-3-4, no velocity, give -1
    for fi in range(frame_num):
        if fi<4 or fi>(frame_num-5): 
            continue
        vects_angles_v[fi]=velocityByframes(vects_angles[fi-4:fi+5])
        #for vbi in range(vect_num):
        #    for vwi in range(vect_num):
        #        vects_angles_v[fi,vbi,vwi]=velocityByframes(vects_angles[fi-4:fi+5,vbi,vwi])
    vects_angles_v=startEndFill(vects_angles_v)
    return(vects_angles_v)


def get_bodyLen_2rats(ratB,ratW):
    len_B=nose2tail_length_median(ratB)
    len_W=nose2tail_length_median(ratW)
    len_body=np.mean([len_B,len_W])
    return(len_body)


def get_near_indexes(ratB,ratW,min_distance_body_times):
    points_dists=get_distances_BetweenPoints(ratB,ratW)
    f_num=points_dists.shape[0]
    len_B=nose2tail_length_median(ratB)
    len_W=nose2tail_length_median(ratW)
    len_body=np.mean([len_B,len_W])
    close_inds=np.zeros(f_num)
    for fi in range(f_num):
        if np.min(points_dists[fi])<=min_distance_body_times*len_body: ##min_distance 
            close_inds[fi]=1
    return(close_inds)



def get_far_indexes(ratB,ratW,min_distance_body_times):
    points_dists=get_distances_BetweenPoints(ratB,ratW)
    f_num=points_dists.shape[0]
    len_B=nose2tail_length_median(ratB)
    len_W=nose2tail_length_median(ratW)
    len_body=np.mean([len_B,len_W])
    close_inds=np.zeros(f_num)
    for fi in range(f_num):
        if np.min(points_dists[fi])>min_distance_body_times*len_body: ##min_distance 
            close_inds[fi]=1
    return(close_inds)


def duration(st):
    ## state labels
    st_num=np.unique(st)
    st_num=st_num[st_num>=0]##
    ##save durations of labels in each file
    label_dict={}
    start_dict={}
    for li in st_num:
        label_dict[str(li)]=[]
        start_dict[str(li)]=[]
    start=0
    for sti in range(1,len(st)):
        if sti==len(st)-1 or st[sti]!=st[start]:
            label_dict[str(st[start])].append(sti-start)
            start_dict[str(st[start])].append(start)
            start=sti
    return([label_dict,start_dict])


def get_close_segments(ratB,ratW,seg_long=15,min_distance_body_times=1):
    ci=get_near_indexes(ratB,ratW,min_distance_body_times)
    label_dict,start_dict=duration(ci)
    ld=label_dict['1.0']
    sd=start_dict['1.0']
    long_inds=np.argwhere(np.array(ld)>=seg_long)
    ldl=np.array(ld)[long_inds]##length of each segment
    sdl=np.array(sd)[long_inds]##starts of each segment
    ##get clips of 15 frames
    starts_frame15=[]
    fps=seg_long
    random.seed(1000)
    for i in range(ldl.shape[0]):
        quot=ldl[i,0]//fps
        wd_size=ldl[i,0]//quot
        ##print('len%s, quot%s, fps-%s, wd_size-%s'%(ldl[i,0],quot,fps,wd_size))
        l_start_inds=[i for i in range(0,ldl[i,0]-fps,wd_size)]
        for si in range(len(l_start_inds)):
            snum=l_start_inds[si]
            if si==(len(l_start_inds)-1):
                rst=random.randint(snum,ldl[i,0]-fps)
                starts_frame15.append(sdl[i,0]+rst)
            else:
                rst=random.randint(snum,snum+wd_size-fps+1)
                starts_frame15.append(sdl[i,0]+rst)
    return(starts_frame15)


def get_far_segments(ratB,ratW,seg_long=15,min_distance_body_times=1):
    ci=get_far_indexes(ratB,ratW,min_distance_body_times)
    label_dict,start_dict=duration(ci)
    ld=label_dict['1.0']
    sd=start_dict['1.0']
    long_inds=np.argwhere(np.array(ld)>=seg_long)
    ldl=np.array(ld)[long_inds]##length of each segment
    sdl=np.array(sd)[long_inds]##starts of each segment
    ##get clips of 15 frames
    starts_frame15=[]
    fps=seg_long
    random.seed(1000)
    for i in range(ldl.shape[0]):
        quot=ldl[i,0]//fps
        wd_size=ldl[i,0]//quot
        ##print('len%s, quot%s, fps-%s, wd_size-%s'%(ldl[i,0],quot,fps,wd_size))
        l_start_inds=[i for i in range(0,ldl[i,0]-fps,wd_size)]
        for si in range(len(l_start_inds)):
            snum=l_start_inds[si]
            if si==(len(l_start_inds)-1):
                rst=random.randint(snum,ldl[i,0]-fps)
                starts_frame15.append(sdl[i,0]+rst)
            else:
                rst=random.randint(snum,snum+wd_size-fps+1)
                starts_frame15.append(sdl[i,0]+rst)
    return(starts_frame15)


def cluster_plot2(embedding,classes,pathFig,centers=None):
    plt.figure(figsize=(6, 6),dpi=300)
    plt.scatter(x=embedding[:, 0], y=embedding[:, 1],c=classes, cmap='brg', s=10)  #jet,Spectral
    if centers:
        plt.scatter(centers[:,0], centers[:,1], marker="v", picker=True)
    plt.gca().set_aspect('equal', 'datalim')
    ##
    cMax=np.max(classes)+1
    plt.colorbar(boundaries=np.arange(cMax)).set_ticks(np.arange(cMax))
    plt.xlabel('UMAP-PC1')
    plt.ylabel('UMAP-PC2')
    plt.tight_layout()
    plt.savefig(pathFig)
    #plt.close()

def cluster_plot(embedding,classes,pathFig):
    plt.figure(figsize=(10, 10),dpi=300)
    plt.scatter(x=embedding[:, 0], y=embedding[:, 1],c=classes, cmap='Spectral', s=4)
    plt.gca().set_aspect('equal', 'datalim')
    ##
    #cMax=np.max(classes)+1
    plt.colorbar()#boundaries=np.arange(cMax)).set_ticks(np.arange(cMax))
    plt.xlabel('UMAP-PC1')
    plt.ylabel('UMAP-PC2')
    plt.tight_layout()
    plt.savefig(pathFig)
    plt.close()


def plot_knn_decision_boundary(X, y, pathFig,k=1):
    #plt.close('all')
    # data setting
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # meshgrid的shape是:[第二个输入的.shape[0], 第一个输出的.shape[0]]
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # x.ravel就是flatten(). 然后np.c_就是按列排
    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X,y)
    z = knn.predict(np.c_[xx.ravel(), yy.ravel()])  # shape:[meshgrid两个shape相乘，2]
    z = z.reshape(xx.shape)  # [x*y] -> [x, y]. 这里z就是对应点的分类效果
    # levels显示的区域
    plt.contour(xx, yy, z,  cmap=plt.cm.Spectral)  # , cmp=plt.cm.Spectral or colors=['red']
    #plt.contour(xx, yy, z, levels=[10, 30, 50], cmap=plt.cm.brg)  # , cmp=plt.cm.Spectral or colors=['red']
    # plt.contour(xx, yy, y.reshape(xx.shape), levels=[0.5], colors=['blue'])
    #plt.show()
    plt.savefig(pathFig)



def plot_knn_decision_boundary_scatter(X, y, pathFig,k=1):
    #k=200
    plt.figure(figsize=(10, 10),dpi=300)
    #plt.close('all')
    # data setting
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # meshgrid的shape是:[第二个输入的.shape[0], 第一个输出的.shape[0]]
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # x.ravel就是flatten(). 然后np.c_就是按列排
    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X,y)
    xxyy=np.c_[xx.ravel(), yy.ravel()]
    z0 = knn.predict(xxyy)  # shape:[meshgrid两个shape相乘，2]
    z = z0.reshape(xx.shape)  # [x*y] -> [x, y]. 这里z就是对应点的分类效果
    zc=z+1  ##from 2 to differ from scatter
    # add mosaik
    plt.contourf(xx,yy,zc,cmap=plt.cm.Spectral)  #mosiak
    ##add contour
    cNOs=list(set(y))
    #plt.contour(xx, yy, z,levels=cNOs)  # , cmp=plt.cm.Spectral or colors=['red']
    c=plt.contour(xx, yy, zc,levels=list(set(zc.ravel())))  # , cmp=plt.cm.Spectral or colors=['red']
    #plt.clabel(c,inline=True,fontsize=15)
    
    ##add cluster number finally
    centers=np.stack([np.quantile(xxyy[np.where(z0==i)],0.5,axis=0) for i in cNOs])
    for i,c in enumerate(cNOs):
      #if c==4:continue
      plt.text(centers[i,0],centers[i,1],str(c),fontsize=15)
    #plt.text(2.5,6.8,'1',fontsize=15)
    #plt.text(-1.6,6.2,'3',fontsize=15)
    #plt.text(1,4.2,'4',fontsize=15)
    #plt.text(2.9,5.2,'4',fontsize=15)
    #plt.text(-1,9,'9',fontsize=15)
    #plt.text(-0.8,10.3,'19',fontsize=15)
    #plt.text(-3,6.5,'13',fontsize=15)
    ##scatter plot from 1
    plt.scatter(x=X[:, 0], y=X[:, 1],c=y, cmap='Spectral', s=4)
    ##add contour
    plt.gca().set_aspect('equal', 'datalim')
    ##
    cMax=np.max(y)+1
    plt.colorbar(boundaries=np.arange(1,cMax)).set_ticks(np.arange(1,cMax))
    plt.xlabel('UMAP-PC1')
    plt.ylabel('UMAP-PC2')
    plt.tight_layout()
    plt.savefig(pathFig)
    plt.close()



def plot_knn_decision_boundary_mosaik(X, y, pathFig,k=1):
    #k=200
    plt.figure(figsize=(10, 10),dpi=300)
    #plt.close('all')
    # data setting
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # meshgrid的shape是:[第二个输入的.shape[0], 第一个输出的.shape[0]]
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # x.ravel就是flatten(). 然后np.c_就是按列排
    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X,y)
    xxyy=np.c_[xx.ravel(), yy.ravel()]
    z0 = knn.predict(xxyy)  # shape:[meshgrid两个shape相乘，2]
    z = z0.reshape(xx.shape)  # [x*y] -> [x, y]. 这里z就是对应点的分类效果
    zc=z  ##from 2 to differ from scatter
    # add mosaik
    plt.contourf(xx,yy,zc,cmap=plt.cm.Spectral)  #mosiak
    ##add contour
    cNOs=list(set(y))
    #plt.contour(xx, yy, z,levels=cNOs)  # , cmp=plt.cm.Spectral or colors=['red']
    c=plt.contour(xx, yy, zc,levels=list(set(zc.ravel())))  # , cmp=plt.cm.Spectral or colors=['red']
    #plt.clabel(c,inline=True,fontsize=15)
    ##add cluster number finally with manual check
    centers=np.stack([np.quantile(xxyy[np.where(z0==i)],0.5,axis=0) for i in cNOs])
    cNOs_dicts={}
    for i in cNOs: #from 1
        cNOs_dicts[i]=centers[i-1]
    ##manual modification!!!
    cNOs_dicts[2][1]=5.5 #xy coordinates
    cNOs_dicts[8][1]=3.6
    cNOs_dicts[9][1]=0.5
    cNOs_dicts[13][0]=8.1
    cNOs_dicts[13][1]=3.7
    cNOs_dicts[16][0]=5.2
    cNOs_dicts[16][1]=3
    cNOs_dicts[20][1]=1.2
    cNOs_dicts[22][0]=9.2
    cNOs_dicts[31]=np.array([9.4,2.5])
    cNOs_dicts[37][1]=3.5
    cNOs_dicts[37][0]=7
    cNOs_dicts[38][1]=8.4
    #print(cNOs_dicts)
    ##
    for i,c in enumerate(cNOs):
      plt.text(cNOs_dicts[c][0],cNOs_dicts[c][1],str(c),fontsize=15)
    plt.gca().set_aspect('equal', 'datalim')
    ##
    cMax=np.max(y)+1
    #plt.colorbar(boundaries=np.arange(1,cMax)).set_ticks(np.arange(1,cMax))
    plt.xlabel('UMAP-PC1')
    plt.ylabel('UMAP-PC2')
    plt.tight_layout()
    plt.savefig(pathFig)
    plt.close()





def angle_project2_vectors(vector1,vector2):
    ang=angle_vectors(vector1,vector2)
    ## x-y plane angles
    ang_xy=angle_vectors(vector1[:2],vector2[:2])
    return(np.array([ang,ang_xy]))


def angle_projectXY_vectors(vector1,vector2):
    ang_xy=angle_vectors(vector1[:2],vector2[:2])
    return(ang_xy)




def get_DefVects_between(ratB,ratW):
    ###get defined vects
    f_num=ratB.shape[0]
    vect_num=4
    dim=3
    vects=np.zeros((f_num,vect_num,dim)) #(frame,vect_num=13,3)
    vects[:,0]=ratB[:,0]-ratB[:,4] #body->nose----
    vects[:,1]=ratB[:,0]-ratW[:,4] #bodyW->noseB
    vects[:,2]=ratW[:,0]-ratW[:,4] 
    vects[:,3]=ratW[:,0]-ratB[:,4]
    #for i in range(f_num):
    #    ##head vectors: 30 31 32
    #    vects[i,0]=ratB[i,0]-ratB[i,4] #body->nose----
    #    vects[i,1]=ratB[i,0]-ratW[i,4] #bodyW->noseB
    #    vects[i,2]=ratW[i,0]-ratW[i,4] 
    #    vects[i,3]=ratW[i,0]-ratB[i,4] 
    return(vects)


def get_project2_Angles_between2ratsNB(vects):
    f_num=vects.shape[0]
    angles=np.zeros((f_num,4))
    for fi in range(f_num):
        angles[fi,:2]=angle_project2_vectors(vects[fi,0],vects[fi,1])
        angles[fi,2:4]=angle_project2_vectors(vects[fi,2],vects[fi,3])
    return(angles)

def get_projectXY_Angles_between2ratsNB(vects):
    f_num=vects.shape[0]
    angles=np.zeros((f_num,2))
    for fi in range(f_num):
        angles[fi,0]=angle_projectXY_vectors(vects[fi,0],vects[fi,1])
        angles[fi,1]=angle_projectXY_vectors(vects[fi,2],vects[fi,3])
    return(angles)

def get_angleXY_tailNectNose_velocity(ratB,fps):
    neckNose=ratB[:,0,:2]-ratB[:,3,:2]
    neckTail=ratB[:,5,:2]-ratB[:,3,:2]
    fn=ratB.shape[0]
    angs=np.array([angle_vectors(neckNose[i],neckTail[i]) for i in range(fn)])
    diffAngs=np.zeros(fn)
    diffAngs[1:]=angs[1:]-angs[:-1]
    diffAngs[0]=diffAngs[1]
    time_gap=1/fps
    angV=diffAngs/time_gap
    return(angV)

###0-180
def nvectAng_3points(p123):
    p1,p2,p3=p123[0],p123[1],p123[2]
    n_vect=np.cross(p2-p1,p3-p1)
    length=np.linalg.norm(n_vect)
    #if length==0:length=1
    cosAngs_Z=np.dot(n_vect,np.array([0,0,1]))/length
    ang=math.acos(cosAngs_Z)*180/math.pi
    return(ang)

def shouderKnee_planeAng(ratB):
    fNum=ratB.shape[0]
    angs=np.zeros((fNum,1))
    planeTurns=[[6,8,12],[8,12,10],[12,10,6],[10,6,8]]
    for fi in range(fNum):
       angs[fi]=np.mean([nvectAng_3points(ratB[fi,planeTurns[0]]),nvectAng_3points(ratB[fi,planeTurns[1]]),nvectAng_3points(ratB[fi,planeTurns[2]]),nvectAng_3points(ratB[fi,planeTurns[3]])])
    return(angs)

def shouderBody_kneeBody_planeAng(ratB):
    fNum=ratB.shape[0]
    angs=np.zeros((fNum,2))
    planeTurns=[[6,8,4],[12,10,4]]
    for fi in range(fNum):
       angs[fi,0]=nvectAng_3points(ratB[fi,planeTurns[0]])
       angs[fi,1]=nvectAng_3points(ratB[fi,planeTurns[1]])
    return(angs)
'''
def get_noseneck_angVel(ratB,fps):
    nnVects=ratB[:,0]-ratB[:,3]
    fn=ratB.shape[0]
    diffAngs=np.zeros((fn,1))
    for i in range(1,fn):diffAngs[i,0]=angle_vectors(nnVects[i-1],nnVects[i])
    diffAngs[0,0]=diffAngs[1,0]
    time_gap=1/fps; #fps=15
    angV=diffAngs/time_gap
    return(angV)
'''

def get_backPalmHeightDiff(ratB,ratW):
    ##
    palmBB=np.array([np.mean(ratB[:,[7,9],2],axis=1),ratB[:,4,2]]).T
    palmBW=np.array([np.mean(ratW[:,[7,9],2],axis=1),ratW[:,4,2]]).T
    return(np.concatenate((palmBB-palmBW,palmBB-palmBW[:,[1,0]]),axis=1))
'''
def get_earBackHeightDiff(ratB,ratW):
    ##
    palmBB=np.array([np.mean(ratB[:,[1,2],2],axis=1),ratB[:,4,2]]).T
    palmBW=np.array([np.mean(ratW[:,[1,2],2],axis=1),ratW[:,4,2]]).T
    return(np.concatenate((palmBB-palmBW,palmBB-palmBW[:,[1,0]]),axis=1))


def get_neckBackHeightDiff(ratB,ratW):
    ##
    palmBB=np.array([ratB[:,3,2],ratB[:,4,2]]).T
    palmBW=np.array([ratW[:,3,2],ratW[:,4,2]]).T
    return(np.concatenate((palmBB-palmBW,palmBB-palmBW[:,[1,0]]),axis=1))
'''
def get_angleXY_tailNose_between(ratB,ratW):
    fn=ratB.shape[0]
    tnB=ratB[:,0,:2]-ratB[:,5,:2]
    tnW=ratW[:,0,:2]-ratW[:,5,:2]
    angs=np.array([angle_vectors(tnB[i],tnW[i]) for i in range(fn)])
    return(angs)

#get overlap of 4 body parts
import numpy as np
from shapely.geometry import Polygon # 多边形

def Cal_area_2poly(data1,data2):
    """
    任意两个图形的相交面积的计算
    :param data1: 当前物体
    :param data2: 待比较的物体
    :return: 当前物体与待比较的物体的面积交集
    """
    poly1 = Polygon(data1).convex_hull # Polygon：多边形对象
    poly2 = Polygon(data2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0 # 如果两多边形不相交
    else:
        inter_area = poly1.intersection(poly2).area # 相交面积
    return inter_area

def get_3parts_downShift(ratB):
    ratB=ratB[:,:,:2]
    ##nose-earL-earR
    headPoly=ratB[:,[0,1,2]] 
    #earL-shoulderL-skThirdL-skThirdR-shoulderR-earR
    skThirdL=ratB[:,6]+(ratB[:,10]-ratB[:,6])/3 #shoulderL-kneeL 1/3
    skThirdR=ratB[:,8]+(ratB[:,12]-ratB[:,8])/3 #shoulderR-kneeR 1/3
    neckPoly=np.transpose(np.array([ratB[:,1],ratB[:,6],skThirdL,skThirdR,ratB[:,8],ratB[:,2]]),[1,0,2]) 
    #skCenterL-kneeL-tail-kneeR-skCenterR
    ktCenterL=(ratB[:,10]+ratB[:,5])/2
    ktCenterR=(ratB[:,12]+ratB[:,5])/2
    bellyPoly=np.transpose(np.array([skThirdL,ratB[:,10],ktCenterL,ratB[:,5],ktCenterR,ratB[:,12],skThirdR]),[1,0,2])
    return([headPoly,neckPoly,bellyPoly])

def get_3partsHeight_downShift(ratB):
    ratB=ratB[:,:,2]     ##frame,point_num   #####Height
    headH=np.mean(ratB[:,[0,1,2]],axis=1)
    skThirdL=ratB[:,6]+(ratB[:,10]-ratB[:,6])/3 #shoulderL-kneeL 1/3
    skThirdR=ratB[:,8]+(ratB[:,12]-ratB[:,8])/3 #shoulderR-kneeR 1/3
    ktCenterL=(ratB[:,10]+ratB[:,5])/2
    ktCenterR=(ratB[:,12]+ratB[:,5])/2
    neckH=np.mean(np.array([ratB[:,1],ratB[:,6],skThirdL,skThirdR,ratB[:,8],ratB[:,2]]),axis=0)
    bellyH=np.mean(np.array([skThirdL,ratB[:,10],ktCenterL,ratB[:,5],ktCenterR,ratB[:,12],skThirdR]),axis=0)
    return(np.array([headH,neckH,bellyH]))

def get_overlap3parts_sign_downShift(ratB,ratW):
    fn=ratB.shape[0]
    overlaps=np.zeros((fn,3,3))
    ##3 polygons
    ratB3parts=get_3parts_downShift(ratB)
    ratW3parts=get_3parts_downShift(ratW)
    ##Mean Heights of 3 polygons
    ratB3partsH=get_3partsHeight_downShift(ratB)
    ratW3partsH=get_3partsHeight_downShift(ratW)
    for fi in range(fn):
      for i in range(3):
        for j in range(3):
          overlaps[fi,i,j]=Cal_area_2poly(ratB3parts[i][fi],ratW3parts[j][fi])*np.sign(ratB3partsH[i,fi]-ratW3partsH[j,fi])
    return(overlaps)   


def get_3parts(ratB):
    ratB=ratB[:,:,:2]   
    ###nose-earL-shoulderL-sholderR-earR
    headPoly=ratB[:,[0,1,6,8,2]]   
    skCenterL=(ratB[:,6]+ratB[:,10])/2
    skCenterR=(ratB[:,8]+ratB[:,12])/2
    shPoly=np.transpose(np.array([ratB[:,6],skCenterL,skCenterR,ratB[:,8]]),[1,0,2]) #sholderL-skCenterL-skCenterR-shoulderR
    knPoly=np.transpose(np.array([skCenterL,ratB[:,10],ratB[:,5],ratB[:,12],skCenterR]),[1,0,2]) #skCenterL-kneeL-tail-kneeR-skCenterR
    return([headPoly,shPoly,knPoly])


def get_3partsHeight(ratB):
    ratB=ratB[:,:,2]     ##frame,point_num   #####Height
    #print(ratB.shape)
    headPolyH=np.mean(ratB[:,[0,1,6,8,2]],axis=1)
    #
    skCenterL=(ratB[:,6]+ratB[:,10])/2
    skCenterR=(ratB[:,8]+ratB[:,12])/2
    shPolyH=np.mean(np.array([ratB[:,6],skCenterL,skCenterR,ratB[:,8]]),axis=0)
    knPolyH=np.mean(np.array([skCenterL,ratB[:,10],ratB[:,5],ratB[:,12],skCenterR]),axis=0)
    return(np.array([headPolyH,shPolyH,knPolyH]))

def get_overlap3parts_sign(ratB,ratW):
    fn=ratB.shape[0]
    overlaps=np.zeros((fn,3,3))
    ##3 polygons
    ratB3parts=get_3parts(ratB)
    ratW3parts=get_3parts(ratW)
    ##Mean Heights of 3 polygons
    ratB3partsH=get_3partsHeight(ratB)
    ratW3partsH=get_3partsHeight(ratW)
    for fi in range(fn):
      for i in range(3):
        for j in range(3):
          overlaps[fi,i,j]=Cal_area_2poly(ratB3parts[i][fi],ratW3parts[j][fi])*np.sign(ratB3partsH[i,fi]-ratW3partsH[j,fi])
    return(overlaps)   

def get_overlap3parts_sign2(ratB,ratW):
    fn=ratB.shape[0]
    overlaps=np.zeros((fn,5))
    ##3 polygons
    ratB3parts=get_3parts(ratB)
    ratW3parts=get_3parts(ratW)
    ##Mean Heights of 3 polygons
    ratB3partsH=get_3partsHeight(ratB)
    ratW3partsH=get_3partsHeight(ratW)
    for fi in range(fn):
        i=0;j=1
        overlaps[fi,0]=Cal_area_2poly(ratB3parts[i][fi],ratW3parts[j][fi])*np.sign(ratB3partsH[i,fi]-ratW3partsH[j,fi])
        i=1;j=0
        overlaps[fi,1]=Cal_area_2poly(ratB3parts[i][fi],ratW3parts[j][fi])*np.sign(ratB3partsH[i,fi]-ratW3partsH[j,fi])
        i=1;j=1
        overlaps[fi,2]=Cal_area_2poly(ratB3parts[i][fi],ratW3parts[j][fi])*np.sign(ratB3partsH[i,fi]-ratW3partsH[j,fi])
        i=1;j=2
        overlaps[fi,3]=Cal_area_2poly(ratB3parts[i][fi],ratW3parts[j][fi])*np.sign(ratB3partsH[i,fi]-ratW3partsH[j,fi])
        i=2;j=1
        overlaps[fi,4]=Cal_area_2poly(ratB3parts[i][fi],ratW3parts[j][fi])*np.sign(ratB3partsH[i,fi]-ratW3partsH[j,fi])
    return(overlaps)   


def get_NoseTail_zone(ratB,i0,j1,j2,length_ratio):
    #length_ratio=1 #1, nose, tail 0.5
    ##nose:0,earL:1,earR:2; #tail:5,kneeL:10,kneeR:12
    ratB=ratB[:,:,:2]
    center=ratB[:,i0]
    dist_noseEar = np.mean([np.mean(np.sqrt(np.sum((center-ratB[:,j1])**2, axis=1))),np.mean(np.sqrt(np.sum((center-ratB[:,j2])**2, axis=1)))])
    r=dist_noseEar*length_ratio
    ##get circle points
    xys=np.array([[center[:,0]+r*np.sin(ang),center[:,1]+r*np.cos(ang)] for ang in np.arange(0,2*np.pi,30*np.pi/180)])
    nosePoly=np.transpose(xys, [2,0,1]) 
    return(nosePoly)

def get_overlap_NoseTail_zone(ratB,ratW):
    fn=ratB.shape[0]
    overlaps=np.zeros((fn,3))
    ##nose zone, tail zone
    ratB2parts=[get_NoseTail_zone(ratB,0,1,2,0.5),get_NoseTail_zone(ratB,5,10,12,0.5)]
    ratW2parts=[get_NoseTail_zone(ratW,0,1,2,0.5),get_NoseTail_zone(ratW,5,10,12,0.5)]
    for fi in range(fn):
      i=0;j=0
      overlaps[fi,0]=Cal_area_2poly(ratB2parts[i][fi],ratW2parts[j][fi])
      i=0;j=1
      overlaps[fi,1]=Cal_area_2poly(ratB2parts[i][fi],ratW2parts[j][fi])
      i=1;j=0
      overlaps[fi,2]=Cal_area_2poly(ratB2parts[i][fi],ratW2parts[j][fi])
    return(overlaps)   

def get_overlap_NoseTail_zone_sign(ratB,ratW):
    fn=ratB.shape[0]
    overlaps=np.zeros((fn,3))
    ##0 nose zone, 5 tail zone
    ratB2parts=[get_NoseTail_zone(ratB,0,1,2,0.5),get_NoseTail_zone(ratB,5,10,12,0.5)]
    ratW2parts=[get_NoseTail_zone(ratW,0,1,2,0.5),get_NoseTail_zone(ratW,5,10,12,0.5)]
    ##zone overlap * sign
    for fi in range(fn):
      i=0;j=0;i2=0;j2=0
      overlaps[fi,0]=Cal_area_2poly(ratB2parts[i][fi],ratW2parts[j][fi])*np.sign(ratB[fi,i2,2]-ratW[fi,j2,2])
      i=0;j=1;i2=0;j2=5
      overlaps[fi,1]=Cal_area_2poly(ratB2parts[i][fi],ratW2parts[j][fi])*np.sign(ratB[fi,i2,2]-ratW[fi,j2,2])
      i=1;j=0;i2=5;j2=0
      overlaps[fi,2]=Cal_area_2poly(ratB2parts[i][fi],ratW2parts[j][fi])*np.sign(ratB[fi,i2,2]-ratW[fi,j2,2])
    return(overlaps)
