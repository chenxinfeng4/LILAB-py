#conda activate mmdet
##after segmentation and merge
# %%python -m lilab.mmdet.t1_finetune_extract_overlap_frames_random $video_path #--iou_th 0.3 --extract_frame_num 30
# ls *.segpkl | sed 's/.segpkl/.mp4/' | xargs -n 1 -P 4 -I {} python -m lilab.mmdet.t1_finetune_extract_overlap_frames_random {} --iou_th 0.5 --extract_frame_num 100
import pickle,cv2,os
import pycocotools._mask as mask_util
import os.path as osp
import numpy as np
import argparse
import tqdm
from random import sample


##iou between two boxes
def bb_iou_xywh(boxA, boxB):
    #boxA = [int(x) for x in boxA] ##xywh
    #boxB = [int(x) for x in boxB]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    iou = interArea / float(boxA[2] * boxA[3]  + boxB[2] * boxB[3] - interArea)
    return iou


def extraction(args):
    video=args.video_path
    #video='/DATA/chenxinfeng/tao_rat/data/20220613-side6-addition/TPH2-KO-multiview-202201/male-test/bwt-wwt-01-18_12-54-05_5min.mp4'
    ##get segmentation overlap
    segpkl=video.replace('.mp4','.segpkl')
    data = pickle.load(open(segpkl, 'rb'))['segdata']
    #calculate bbox iou of each view
    iou_th=args.iou_th  ##smaller, more
    #iou_th=0.3
    iou_views6=np.zeros((len(data),len(data[0]))) #from view 0 to view 5
    for iview in range(len(data)): 
        for fi,label in enumerate(data[iview]): #frame
            seg = label[1]  ##0--box, 1--mask
            if len(seg)==2: 
                if len(seg[0])==0 or len(seg[1])==0: continue
                iou=round(bb_iou_xywh(mask_util.toBbox(seg[0])[0],mask_util.toBbox(seg[1])[0]),2)  #black and white ##xywh
                if iou>=iou_th:
                    iou_views6[iview][fi]=1
    ##get multi-view overlapped frames
    over_view_num=3
    iou_views6_sum=np.sum(iou_views6,axis=0) ##for each frame
    overlap_frame_NOs=np.argwhere(iou_views6_sum>=over_view_num)[:,0]
    #times=['min:sec=%s:%s'%(int(fi/30//60),np.round(fi/30%60,2)[0]) for fi in overlap_frame_NO]

    ##get continuous overlap frames, max time gap gap_sec
    gap_sec=0.5 #0.5~1 maybe, smaller, more segments
    cap = cv2.VideoCapture(video);gap_frames=gap_sec*cap.get(cv2.CAP_PROP_FPS);
    ##gap between two neighbouring frames
    fn_diff=overlap_frame_NOs[1:]-overlap_frame_NOs[:-1]
    seg_NO_ends=np.insert(np.argwhere(fn_diff>=gap_frames)[:,0]+1,0,0) ##0 end0,end1 end2
    seg_frame_NOs=[list(overlap_frame_NOs[seg_NO_ends[si-1]:seg_NO_ends[si]]) for si in range(1,len(seg_NO_ends))]
    ##random choose num_choose segments, random choose 1 from each segments
    #num_choose_overlap=30
    num_choose_overlap=args.extract_frame_num
    num_choose=min(num_choose_overlap,len(seg_frame_NOs))
    frame_chooses=[sample(sfs,1)[0] for sfs in sample(seg_frame_NOs,num_choose)]
    ##save each frame
    path = osp.dirname(video)
    frame_path=osp.join(path,'frames_overlap')
    os.makedirs(frame_path,exist_ok=True)
    ##
    videoName=os.path.split(video)[-1]
    for fc in np.sort(frame_chooses):
        print(fc)
        cap.set(cv2.CAP_PROP_POS_FRAMES,fc)
        _,img=cap.read()
        img_name=osp.join(frame_path,videoName+'_boxoverlapRate%s-frame%06d.jpg'%(iou_th,fc))
        cv2.imwrite(img_name,img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, default=None, help='path to video or folder')
    parser.add_argument('--iou_th', type=float, default=0.3)
    parser.add_argument('--extract_frame_num', type=int, default=30)
    args = parser.parse_args()
    extraction(args)