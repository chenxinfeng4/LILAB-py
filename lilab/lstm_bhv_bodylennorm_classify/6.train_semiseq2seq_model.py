# conda activate OpenLabCluster
#%%
import numpy as np
import torch
import torch.nn as nn
import os
import os.path as osp
from torch.utils.data import DataLoader, TensorDataset
import tqdm
from openlabcluster.training_utils.ssl.SeqModel import SemiSeq2Seq
from openlabcluster.utils import auxiliaryfunctions
import lilab.OpenLabCluster_train.model
from torch import optim
from pathlib import Path
import sys
import pickle
from openlabcluster.training_utils.ssl.seq_train import training

"""
is_percentile_dependent: 是否已经对 速度特征启用 95% 分位数归一化
"""
is_percentile_dependent = True

project = '/DATA/taoxianming/rat/data/Mix_analysis/SexAgeDay55andzzcWTinAUT_MMFF/result32/olc-iter4-2024-05-27'
model_path0 = '/DATA/taoxianming/rat/data/Mix_analysis/SexAgeDay55andzzcWTinAUT_MMFF/result32/olc-iter1-2024-05-23/models/FWPCA0.00_P100_en3_hid30_epoch2' 
class FAKE: pass
cfg = osp.join(project, 'config.yaml')
cfg_data = auxiliaryfunctions.read_config(cfg)
self = FAKE()
self.cfg = cfg
self.model_name = model_path0
self.cfg_data = cfg_data


batch_size = 64
hidden_size = self.cfg_data['hidden_size']
en_num_layers = self.cfg_data['en_num_layers']
de_num_layers = self.cfg_data['de_num_layers']
cla_num_layers = self.cfg_data['cla_num_layers']
fix_state = self.cfg_data['fix_state']
teacher_force = self.cfg_data['teacher_force']
device = 'cuda:0'


#%% load dataset
clip_label_data = pickle.load(open('/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day55_Mix_analysis/SexAgeDay55andzzcWTinAUT_MMFF/data/train_test_clip_newlabels_k36.pkl','rb'))
out_res_pkl_file = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day55_Mix_analysis/SexAgeDay55andzzcWTinAUT_MMFF/data/lstm_result_maskdannce_feat32_try1.pkl'
df_records = clip_label_data['df_records']

feat_name = 'feat_clip_norm' if is_percentile_dependent else 'feat_clip'
ind_train = df_records['isTrain']
train_X = np.array([*df_records[feat_name].values[ind_train]])
train_y = df_records['cluster_id'].values[ind_train] - df_records['cluster_id'].min() + 1 # start 1

ind_test = ~df_records['isTrain']
test_X = np.array([*df_records[feat_name].values[ind_test]])
test_y = df_records['cluster_id'].values[ind_test] - df_records['cluster_id'].min() + 1 #start 1

train_dataset = TensorDataset(torch.from_numpy(train_X.transpose(0, 2, 1)),
                              torch.ones(len(train_X), dtype=torch.long)*24,
                              torch.from_numpy(train_y),
                              torch.from_numpy(train_y),
                              torch.from_numpy(train_y),
                              )
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(torch.from_numpy(test_X.transpose(0, 2, 1)),
                             torch.ones(len(test_X), dtype=torch.long)*24,
                             torch.from_numpy(test_y),
                             torch.from_numpy(test_y),
                             torch.from_numpy(test_y)
                             )
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

nsample, feature_length, seq_length = train_X.shape
assert seq_length == 24 and nsample > 1000

phase = 'PC'
fix_weight = False

if fix_weight:
    network = 'FW' + phase

if fix_state:
    network = 'FS' + phase

if not fix_state and not fix_weight:
    network = 'O' + phase

# hyperparameter
learning_rate = self.cfg_data['learning_rate']
epoch = 10   #训练2-3轮

cla_dim = self.cfg_data['cla_dim'] = [train_y.max()]
model:nn.Module = SemiSeq2Seq(feature_length, hidden_size, feature_length, batch_size,
                                cla_dim, en_num_layers, de_num_layers, cla_num_layers, fix_state, fix_weight, teacher_force, device).to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)


criterion_seq = nn.L1Loss(reduction='none')
criterion_cla = nn.CrossEntropyLoss(reduction='sum')

alpha = 0.1
lambda1 = lambda ith_epoch: 0.95 ** (ith_epoch)
model_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
past_loss = sys.float_info.max
self.hidden_size = hidden_size
self.num_class = num_class = train_y.max()
self.alpha = alpha
self.few_knn = False
self.device = device
print_every = 1
self.canvas = None


#%%

output_path = '/home/liying_lab/chenxf/ml-project/LILAB-py/lilab/lstm_bhv_bodylennorm_classify/output' + ('_percDepent' if is_percentile_dependent else '')
model_path = '/home/liying_lab/chenxf/ml-project/LILAB-py/lilab/lstm_bhv_bodylennorm_classify/model'   + ('_percDepent' if is_percentile_dependent else '')
os.makedirs(output_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

percentage = 1
file_output = open(os.path.join(output_path, '%sA%.2f_P%d_en%d_hid%d.txt' % (
    network, alpha, percentage * 100, en_num_layers, hidden_size)), 'w')
file_test_output = open(os.path.join(output_path, '%sA%.2f_P%d_en%d_hid%d_test.txt' % (
    network, alpha, percentage * 100, en_num_layers, hidden_size)), 'w')
model_prefix = os.path.join(model_path, '%sA%.2f_P%d_en%d_hid%d' % (
    network, alpha, percentage * 100, en_num_layers, hidden_size))
few_knn = False
model_path = Path(model_prefix).parent
pre = Path(model_prefix).name
k = 2 


def testing(file_test_output):
    pred_label_l = []
    semi_label_l = []
    cla_loss_l = []
    seq_loss_l = []
    for data, seq_len, _, semi_label, _ in tqdm.tqdm(test_dataloader):
        input_tensor = data.to(device)
        semi_label = torch.tensor(semi_label, dtype=torch.long).to(device)
        with torch.no_grad():
            en_hi, de_out, cla_pre = model(input_tensor, seq_len)
            pred_label_l.extend(cla_pre.argmax(1).tolist())
            semi_label_l.extend((semi_label-1).tolist())
            label = semi_label
            if sum(label != 0) != 0:
                cla_loss = criterion_cla(cla_pre[label != 0], label[label != 0] - 1)
            else:
                cla_loss = 0

            mask = torch.zeros([len(seq_len), max(seq_len)]).to(device)
            for ith_batch in range(len(seq_len)):
                mask[ith_batch, 0:seq_len[ith_batch]] = 1
            mask = torch.sum(mask, 1)

            seq_loss = torch.sum(criterion_seq(de_out, input_tensor), 2)
            seq_loss = torch.mean(torch.sum(seq_loss, 1) / mask)
            cla_loss_l.append(cla_loss.item())
            seq_loss_l.append(seq_loss.item())

    seq = np.mean(seq_loss_l)
    cla = np.mean(cla_loss_l)

    pred_label_l = np.array(pred_label_l)
    semi_label_l = np.array(semi_label_l)
    acc_test = np.sum(pred_label_l == semi_label_l) / np.sum(semi_label_l >=0)

    file_test_output.write(f"{seq:.3f} {cla:.3f} {acc_test:.3f}\n")
    print(f"Test clas loss: {cla:.3f} seq_loss:{seq:.3f} acc:{acc_test:.3f}")
    return acc_test


ith_epoch = 0
for ith_epoch in range(epoch):
    past_loss, model_name, self.acc = training(ith_epoch, epoch, train_loader, print_every, self.canvas,
            model, optimizer, criterion_seq, criterion_cla, alpha, k, file_output, past_loss,model_path, pre,
            hidden_size, model_prefix, num_class,
            few_knn, device)
    acc_test = testing(file_test_output)
    model_scheduler.step()
    file_output.flush()
    file_test_output.flush()

#%%
class SemiSeq2Seq_simple(nn.Module):
    def __init__(self, model:SemiSeq2Seq):
        super().__init__()
        self.model = model
    
    def forward(self, input_tensor):
        seq_len = [24]
        *_, pred = self.model(input_tensor, seq_len)
        return pred

#%%
model_simple = SemiSeq2Seq_simple(model)
input_tensor = torch.randn(1, 24, 32).to(device).float()
pred = model_simple(input_tensor)
model_simple.eval()

pt_filename = f"{model_path}/semiseq2seq_behavior_offline.pt" 
torch.save(model_simple, pt_filename)
print('Saved model to %s' % pt_filename)