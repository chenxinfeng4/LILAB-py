from multiprocessing import Queue
import numpy as np
import sys
import pickle
import torch
from torch2trt import TRTModule
from scipy.signal import medfilt

sys.path.append('/home/liying_lab/chenxf/ml-project/论文图表/yolo_dannce_训练')
from utilities_package_feature import package_feature, np_norm


pvalue_thr = 0.9

engine = '/home/liying_lab/chenxf/ml-project/论文图表/yolo_dannce_训练/lstm_behavior.engine'
cluster_names_pkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day55_Mix_analysis/Day55_yolo_rt/new_cluster_name.pkl'

class RecentClip:
    def __init__(self, clip_windowsize):
        self.clip_windowsize = clip_windowsize + 3
        self.clip_buffer = np.empty((1,))
        self.clip_cursor:int = 0
        self.p3d_smooth = np.empty((1,))
        self.waitInit = True
        
    def push_getclip(self, data_aframe:np.ndarray) -> np.ndarray:
        if self.waitInit:
            self.waitInit = False
            self.clip_buffer = np.zeros([self.clip_windowsize, *data_aframe.shape]) + data_aframe[None]
            self.p3d_smooth = np.zeros_like(self.clip_buffer)
        self.clip_cursor = (self.clip_cursor + 1) % self.clip_windowsize
        self.clip_buffer[self.clip_cursor] = data_aframe
        out = np.concatenate([self.clip_buffer[self.clip_cursor+1:], self.clip_buffer[:self.clip_cursor+1]])
        return out

    def push_getsmooth(self, data_p3d:np.ndarray) -> np.ndarray:
        out_array = self.push_getclip(data_p3d)
        out_array_ravel = out_array.reshape(self.clip_windowsize, -1)
        p3d_smooth_ravel = self.p3d_smooth.reshape(self.clip_windowsize, -1)
        for i in range(out_array_ravel.shape[1]):
            p3d_smooth_ravel[:,i] = medfilt(out_array_ravel[:,i], 7)
        return self.p3d_smooth[3:]

def msg_fatory(cluster_names_pkl:str):
    from lilab.yolo_seg.sockerServer import port
    import picklerpc
    rpc_client = picklerpc.Client(('127.0.0.1', port))
    print('[Cluster] Connect to RPC.', rpc_client.label_str())

    cluster_names = pickle.load(open(cluster_names_pkl, 'rb'))['new_cluster_name']
    ncluster = len(cluster_names)
    
    def call(clusterid:int, pvalue:float):
        if pvalue < pvalue_thr: return
        if clusterid < 0 or clusterid >= ncluster: return
        label_str:str = cluster_names[clusterid]
        rpc_client.label_str(label_str)

    return call

def softmax(x):
    exp_x = np.exp(x)
    return(exp_x/exp_x.sum())

def cluster_main(q:Queue, model_smooth_matcalibpkl:str):
    # numpy_p3d = np.frombuffer(share_array_p3d.get_obj(), dtype=np.float64).reshape((NFRAME, *p3d_numpy_shape))
    p3d = pickle.load(open(model_smooth_matcalibpkl, 'rb'))['keypoints_xyz_ba']
    p3d_CSK3 = p3d.transpose([1,0,2,3]).astype(float)

    body_length = np.mean(np.median(np_norm(p3d_CSK3[:,:,0] - p3d_CSK3[:,:,5]), axis= -1))
    sniff_zoom_length = np.mean(np.median(np_norm(p3d_CSK3[:,:,0,:2] - p3d_CSK3[:,:,1,:2]), axis= -1))/2
    
    torch.cuda.set_device("cuda:0")
    trt_model = TRTModule()
    trt_model.load_from_engine(engine)
    input_shape = tuple(trt_model.context.get_binding_shape(0))
    input_numpy = np.random.rand(*input_shape).astype(np.float32)
    output = trt_model(torch.from_numpy(input_numpy).cuda().float())
    
    recentClip = RecentClip(24)
    msg_logger = msg_fatory(cluster_names_pkl)
    result_l = []
    while True:
        p3d = q.get()
        if p3d is None: break

        p3d_clip = recentClip.push_getsmooth(p3d).transpose(1,0,2,3) #(2,24,14,3)
        out_feature = package_feature(p3d_clip, body_length, sniff_zoom_length)  #(34,24)
        out_feature_torch = torch.from_numpy(out_feature.T[None]).cuda().float() #(1,24,34)
        out_label = trt_model(out_feature_torch)[0].detach().cpu().numpy().squeeze()
        ind_max = np.argmax(out_label)
        pval = softmax(out_label)[ind_max]
        msg_logger(ind_max, pval)
        result_l.append((ind_max, pval))

    pickle.dump(result_l, open('/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/test/out.rtclusterpkl', 'wb'))
    print('[3] Cluster done')
