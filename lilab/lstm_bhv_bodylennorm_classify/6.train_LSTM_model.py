#%%
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import os
from sklearn.metrics import confusion_matrix

"""
0.850 0.968 0.906

"""
is_percentile_dependent = True

clip_label_data = pickle.load(open('/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day55_Mix_analysis/Day55_yolo_rt/train_test_clip_newlabels_k36.pkl','rb'))
out_res_pkl_file = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day55_Mix_analysis/SexAgeDay55andzzcWTinAUT_MMFF/data/lstm_result_maskdannce_feat32_try1.pkl'
df_records = clip_label_data['df_records']

feat_name = 'feat_clip_keep' if False else 'feat_clip'
ind_train = df_records['isTrain']
train_X = np.array([*df_records[feat_name].values[ind_train]])
train_y = df_records['cluster_id'].values[ind_train] - df_records['cluster_id'].min()

ind_test = ~df_records['isTrain']
test_X = np.array([*df_records[feat_name].values[ind_test]])
test_y = df_records['cluster_id'].values[ind_test] - df_records['cluster_id'].min()


#%%
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2  # 双向 LSTM
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * self.num_directions, 64)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]  # 获取最后一个时间步的输出
        output = torch.relu(output)
        output = self.fc1(output)
        output = torch.relu(output)
        output = self.fc(output)
        return output


input_size = train_X.shape[1]
hidden_size = 128
num_layers = 3
num_classes = train_y.max()+1
num_epochs = 80
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型实例
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
model = model.to(device)

# 准备数据
train_dataset = TensorDataset(torch.from_numpy(train_X.transpose(0, 2, 1)), torch.from_numpy(train_y))
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(torch.from_numpy(test_X.transpose(0, 2, 1)), torch.from_numpy(test_y))
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
# 训练模型
for epoch in tqdm.trange(num_epochs):
    train_loss_l = []
    train_accuracy_l = []
    for batch_x, batch_y in train_dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # 计算训练集准确率和loss
        _, predicted = torch.max(outputs.data, 1)
        train_loss_l.append(loss.item())
        train_accuracy_l.append((predicted == batch_y).sum().item() / batch_y.size(0))

    train_loss.append(np.mean(train_loss_l))
    train_accuracy.append(np.mean(train_accuracy_l))

    # 在测试集上评估模型
    test_loss_l = []
    test_accuracy_l = []
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_x, batch_y in test_dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            test_loss_l.append(loss.item())
            test_accuracy_l.append((predicted == batch_y).sum().item() / batch_y.size(0))

    test_loss.append(np.mean(test_loss_l))
    test_accuracy.append(np.mean(test_accuracy_l))
    accuracy = correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss[-1]:.4f}, Accuracy: {train_accuracy[-1]:.4f},  Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

out_model = 'lstm_behavior_offline' + ('_percDependent' if is_percentile_dependent else '')
if True:
    onnx_filename = f"/home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/lstm_bhv_bodylennorm_classify/{out_model}.onnx" 
    torch.onnx.export(model, batch_x[[0]], onnx_filename, verbose=True)
    os.system(f'trtexec --onnx={onnx_filename} --saveEngine={onnx_filename.replace(".onnx",".engine")} --workspace=1024 --fp16')

gt_l = []
pred_l = []

with torch.no_grad():
    correct = 0
    total = 0
    for batch_x, batch_y in test_dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        _, predicted = torch.max(outputs.data, 1)
        
        gt_l.append(batch_y.cpu().numpy())
        pred_l.append(predicted.cpu().numpy())

gt_l = np.concatenate(gt_l)
pred_l = np.concatenate(pred_l)
cm = confusion_matrix(gt_l, pred_l)
pickle.dump({'gt_l':gt_l, 'pred_l':pred_l, 'ind_train':ind_train}, 
            open(out_res_pkl_file, 'wb'))
