#%%
import numpy as np
import itertools
from shapely.geometry import Polygon # 多边形

np_norm = lambda x: np.linalg.norm(x, axis=-1)
body_length = 155
sniff_zoom_length = 15

#%%
def get_point_velocity(point_data:np.ndarray):
    assert point_data.ndim == 3, "point_data should be a 3D array. ANIMALxTimex2"
    cents_v1=np_norm(np.diff(point_data[...,:2],axis=1))
    cents_v = np.concatenate((cents_v1,cents_v1[:,-1:]),axis=1)
    return cents_v


def nvectAng_3points(p123):
    p1,p2,p3=p123[...,0,:],p123[...,1,:],p123[...,2,:]
    n_vect=np.cross(p2-p1,p3-p1,axis=-1)
    length=np.linalg.norm(n_vect,axis=-1) + 0.1
    return n_vect[...,-1] / length

def shouderBody_kneeBody_planeAng(p3d_CTK3):
    C, S, K = p3d_CTK3.shape[:3]
    angs=np.zeros((2,C,S))
    planeTurns=[[6,8,4],[12,10,4]]

    angs[0]=nvectAng_3points(p3d_CTK3[:,:,planeTurns[0]])
    angs[1]=nvectAng_3points(p3d_CTK3[:,:,planeTurns[1]])
    return angs


def get_backPalmHeightDiff(ratB,ratW):
    palmBB=np.array([np.mean(ratB[:,[7,9],2],axis=1),ratB[:,4,2]]).T
    palmBW=np.array([np.mean(ratW[:,[7,9],2],axis=1),ratW[:,4,2]]).T
    return(np.concatenate((palmBB-palmBW,palmBB-palmBW[:,[1,0]]),axis=1))


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
    headPolyH=np.mean(ratB[:,[0,1,6,8,2]],axis=1)
    skCenterL=(ratB[:,6]+ratB[:,10])/2
    skCenterR=(ratB[:,8]+ratB[:,12])/2
    shPolyH=np.mean(np.array([ratB[:,6],skCenterL,skCenterR,ratB[:,8]]),axis=0)
    knPolyH=np.mean(np.array([skCenterL,ratB[:,10],ratB[:,5],ratB[:,12],skCenterR]),axis=0)
    return(np.array([headPolyH,shPolyH,knPolyH]))


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


def get_overlap3parts_sign2(ratB, ratW):
    ratB3parts = get_3parts(ratB)
    ratW3parts = get_3parts(ratW)
    ratB3partsH=get_3partsHeight(ratB)
    ratW3partsH=get_3partsHeight(ratW)
    nsample = len(ratB)
    overlaps = np.zeros((3,3,nsample))
    for isample, i,j in itertools.product(range(nsample), range(3), range(3)):
        overlaps[i, j, isample] = Cal_area_2poly(ratB3parts[i][isample],ratW3parts[j][isample])

    overlaps = overlaps*np.sign(ratB3partsH[:,None]-ratW3partsH[None,:])
    return(overlaps)


def get_overlap_NoseTail_zone2(ratB,ratW,sniff_zoom_length):
    ##ratio is ratius ratio in nose-ear distance, ratio in tail-knee distance
    ratB_nose = ratB[:,0,:2]
    ratB_tail = ratB[:,5,:2]
    ratW_nose = ratW[:,0,:2]
    ratW_tail = ratW[:,5,:2]
    dist = np.array([np_norm(ratB_nose - ratW_nose), np_norm(ratB_nose - ratW_tail), np_norm(ratB_tail - ratW_nose)])
    std = np.ones((3,1), dtype=float)*sniff_zoom_length
    overlaps = gaussian(dist, std)
    return(overlaps)


def gaussian(x, s):
    x2 = np.clip(np.abs(x)-10, 0, None)
    s = np.clip(s-9, 0, None)
    return np.exp(-0.5 * ((x2 - 0) / s)**2)


# body_length = np.mean(np.median(np_norm(p3d_CTK3[:,:,0] - p3d_CTK3[:,:,5]), axis= -1))

def package_feature(p3d_CTK3:np.ndarray, body_length=body_length, sniff_zoom_length=sniff_zoom_length):
    # 1. body speed by back, (2,T)
    out_backSpeed = get_point_velocity(p3d_CTK3[:,:,4]) / body_length

    # 2. body height by back, (2,T)
    out_backHeight = p3d_CTK3[:,:,4,2] / body_length

    # 3. shouder-knee vs palm feet, (2,T)
    out_skpfHdiff=(np.mean(p3d_CTK3[:,:,[6,8,10,12],2],axis=2) -
                np.mean(p3d_CTK3[:,:,[7,9,11,13],2],axis=2)) / body_length

    # 4. planeAng, (4,T)
    out_backAng2=shouderBody_kneeBody_planeAng(p3d_CTK3).reshape(4,-1)

    # 5. heigh diff, (4,T)
    ratB, ratW = p3d_CTK3
    out_backPalmHeightDiff=get_backPalmHeightDiff(ratB, ratW).T / body_length

    # ---1. interaction, (4,T)
    def np_clip20(x): x[x<10] = 0; return x
    nn = np_clip20(np_norm(ratB[:,0,:2] - ratW[:,0,:2])) / body_length
    nt = np_clip20(np_norm(ratB[:,0,:2] - ratW[:,5,:2])) / body_length
    tn = np_clip20(np_norm(ratB[:,5,:2] - ratW[:,0,:2])) / body_length
    bb = np_norm(ratB[:,4,:2] - ratW[:,4,:2]) / body_length
    out_pds = np.array((nn,nt,tn,bb))

    ##---2, one rat's angle.  back-nose-back', (2,T)
    vectors = np.array([[ratB[:,0]-ratB[:,4], #bodyB->noseB----
                        ratB[:,0]-ratW[:,4]], #bodyW->noseB
                        [ratW[:,0]-ratW[:,4], 
                        ratW[:,0]-ratB[:,4]]]) #(Batch,V1V2,Batch,3)
    V1, V2 = vectors[:,0,...,:2], vectors[:,1,...,:2]
    out_angXY = np.arccos(np.sum(V1*V2, axis=-1) / (np_norm(V1) * np_norm(V2)+0.1))

    # ---3, overlap 3x3, (9,T)
    out_overlaps3x3 = get_overlap3parts_sign2(ratB,ratW).reshape(9,-1)
    out_overlaps3x3_norm = out_overlaps3x3 / (body_length**2) * 100

    # ----4 overlap 3
    out_ntZoneOverlap = get_overlap_NoseTail_zone2(ratB,ratW,sniff_zoom_length)

    out_feature = np.concatenate((out_backSpeed,out_backHeight,out_skpfHdiff,
                                out_backAng2,out_backPalmHeightDiff,out_pds,
                                out_angXY,
                                out_overlaps3x3_norm, out_ntZoneOverlap,
                                ),axis=0)
    return out_feature


if __name__ ==  "__main__":
    import pickle
    import matplotlib.pyplot as plt
    import tqdm
    p3d = pickle.load(open('/DATA/taoxianming/rat/data/Mix_analysis/SexAgeDay55andzzcWTinAUT_MMFF/data/2023-09-24_16-39-20G1wWTxH1bWT.smoothed_foot.matcalibpkl', 'rb'))['keypoints_xyz_ba']
    p3d_CTK3 = p3d.transpose([1,0,2,3]).astype(float)

    body_length = np.mean(np.median(np_norm(p3d_CTK3[:,:,0] - p3d_CTK3[:,:,5]), axis= -1))
    sniff_zoom_length = np.mean(np.median(np_norm(p3d_CTK3[:,:,0,:2] - p3d_CTK3[:,:,1,:2]), axis= -1))/2
    for _ in tqdm.trange(1000):
        a = package_feature(p3d_CTK3[:,:24], body_length, sniff_zoom_length)

    ratB, ratW = p3d_CTK3

    # 示例使用
    x = np.linspace(0,50, 100)  # 生成 x 值的范围
    s = 15  # 标准差

    # 计算高斯曲线函数的值
    y = gaussian(x, s)

    # 绘制高斯曲线
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gaussian Curve')
    plt.grid(True)
    plt.show()