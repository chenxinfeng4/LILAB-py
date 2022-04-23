# python -m lilab.smoothpoints.biLSTM_point3d_impute_train
# %% imports
import io
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
import os.path as osp
import torch.nn as nn
import torch.optim as optim
from lilab.smoothpoints.LSTM_base import (test_model, create_datasetloader,
                root_mean_squared_error, root_mean_squared_error3d, root_squared_error3d)

# %% trainingset
# train_files = ['/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_side6/rat/rat_points3d_cm.mat',
#                '/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_side6/15-37-42/white3d/rat_points3d_cm.mat']

train_files = ['/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_30fps/whiteblack/15-11-45-white/rat_points3d_cm.mat',
               #'/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_30fps/whiteblack/15-11-45-black/rat_points3d_cm.mat',
               '/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_30fps/whiteblack/15-37-42-white/rat_points3d_cm.mat',
               #'/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_30fps/whiteblack/15-37-42-black/rat_points3d_cm.mat'
               ]


load_checkpoint = None
# load_checkpoint = '/home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/smoothpoints/modelbilstm.pth'

missing_rate = 0.2
missing_replacevalue= -50
test_rate = 0.082
look_back = look_forward = 36
hidden_size = 60
batch_size = 600
num_layers = 1
num_keypoints = 14
num_dim = 3
n_side = 30
n_epochs = 600
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(0)
torch.manual_seed(0)

# %% models
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Model, self).__init__()
        self.linerinput = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.liner = nn.Linear(hidden_size * 2, output_size)
        self.linertrans = nn.Linear(hidden_size * 2, 3)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        batchsize = x.shape[0]
        device = x.device
        hidden = (torch.zeros(self.num_layers *2, batchsize, self.hidden_size, device=device),
                  torch.zeros(self.num_layers *2, batchsize, self.hidden_size, device=device)) 
        x = self.linerinput(x)
        x = self.relu(x)
        lstmout, _ = self.lstm(x, hidden)
        # lastout = lstmout[:,-1,:]
        lastout1 = lstmout[:,look_back,:self.hidden_size]
        lastout2 = lstmout[:,look_back,self.hidden_size:]
        lastout = torch.cat([lastout1, lastout2], axis=1)
        lastout = self.dropout(lastout)
        out = self.liner(lastout)  #(features,)
        trans = self.linertrans(lastout)  #(3,)
        
        if out.dim() == 1:
            angle, tx, ty = trans
            out_X, out_Y, out_Z = out[::3], out[1::3], out[2::3]
            out_Xtrans = out_X + torch.cos(angle) * (out_X - tx) - torch.sin(angle) * (out_Y-ty)
            out_Ytrans = out_Y + torch.sin(angle) * (out_X - tx) + torch.cos(angle) * (out_Y-ty)
        else:
            angle, tx, ty = trans[:,[0]], trans[:,[1]], trans[:,[2]]
            out_X, out_Y, out_Z = out[:,::3], out[:,1::3], out[:,2::3]
            out_Xtrans = out_X + torch.cos(angle) * (out_X - tx) - torch.sin(angle) * (out_Y-ty)
            out_Ytrans = out_Y + torch.sin(angle) * (out_X - tx) + torch.cos(angle) * (out_Y-ty)
        out[:,::3] = out_Xtrans
        out[:,1::3] = out_Ytrans

        return out
    
# %% data
def create_model(load_checkpoint, device = device):
    input_size = output_size = num_keypoints * num_dim
    model = Model(input_size, hidden_size, num_layers, output_size)
    if load_checkpoint:
        model.load_state_dict(torch.load(load_checkpoint))
    model.to(device)
    print("Map the model to device:", device)
    return model


def train_model(model, train_dataloader):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()

    for epoch in tqdm.tqdm(range(n_epochs)):
        train_losses = []
        for X_batch, y_batch_sparse, y_batch in train_dataloader:        
            X_batch, y_batch_sparse, y_batch = X_batch.to(device), y_batch_sparse.to(device), y_batch.to(device)
            out = model(X_batch)   #(batch, feature)
            pos_nonzero = y_batch != missing_replacevalue
            optimizer.zero_grad()
            loss = mse_loss(out[pos_nonzero], y_batch[pos_nonzero])
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
        
        if epoch % 50 == 0 and epoch>0:
            torch.save(model.state_dict(), osp.join(osp.dirname(__file__), 
            f'model_HD{hidden_size}_BK{look_back}_FW{look_forward}_SD{n_side}_{epoch}.pth'))

        print('Epoch: {}, Loss: {:.4f}'.format(epoch, np.mean(train_losses)))

    out_checkpoint = osp.join(osp.dirname(__file__), 
            f'model_HD{hidden_size}_BK{look_back}_FW{look_forward}_SD{n_side}_final.pth')

    torch.save(model.state_dict(), out_checkpoint)



def main(train_files):
    train_dataloader, test_dataloader = create_datasetloader(train_files, batch_size, 
                        num_keypoints, num_dim, missing_rate,
                        test_rate, n_side, look_back, look_forward)

    # %% create and train the model
    model = create_model(load_checkpoint)
    train_model(model, train_dataloader)

    # %% test the model
    test_sparse_pred, test_sparse_ground_truth, test_dense_ground_truth = test_model(model, test_dataloader)
    
    new_stdout = io.StringIO()
    # sys.stdout = new_stdout

    print('Imputation')
    posImpute = np.where((test_dense_ground_truth != missing_replacevalue) & (test_sparse_ground_truth == missing_replacevalue))
    testPred_rmse = root_mean_squared_error(test_dense_ground_truth, test_sparse_pred, posImpute)
    testPred_cm = root_mean_squared_error3d(test_dense_ground_truth, test_sparse_pred, posImpute)
    print('Test prediction RMSE: %.2f RMSE, %.2f cm' % (testPred_rmse, testPred_cm))

    print('Self fitting')
    posFit = np.where((test_dense_ground_truth != missing_replacevalue) & (test_sparse_ground_truth != missing_replacevalue))
    testPred_rmse = root_mean_squared_error(test_dense_ground_truth, test_sparse_pred, posFit)
    testPred_cm = root_mean_squared_error3d(test_dense_ground_truth, test_sparse_pred, posFit)
    print('Test prediction RMSE: %.2f RMSE, %.2f cm' % (testPred_rmse, testPred_cm))

    if __name__ == '__main__':
        testPred_each_cm = root_squared_error3d(test_dense_ground_truth, test_sparse_pred, posImpute)
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8,8))
        plt.hist(testPred_each_cm, bins=50)
        plt.plot([testPred_cm, testPred_cm], [0, 110], 'k--')
        plt.text(testPred_cm, 110, '%.2f cm' % testPred_cm, ha='right', va='bottom', fontsize=14)
        plt.xlabel('Error (cm)')
        plt.ylabel('Frequence')
        plt.title('Imputation Error')
        

    sys.stdout = sys.__stdout__
    outstring = new_stdout.getvalue()
    print(outstring)
    return outstring


if __name__ == '__main__':
    main(train_files)
