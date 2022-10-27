# %% imports
import io
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io as sio
import os.path as osp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

# %% trainingset

train_files = ['/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_side6/rat/rat_points3d_cm.mat',
               '/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_side6/15-37-42/white3d/rat_points3d_cm.mat']


load_checkpoint = None
# load_checkpoint = '/home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/smoothpoints/model.pth'


missing_rate = 0.2
missing_replacevalue= -50
test_rate = 0.082
look_back = 24
hidden_size = 60
num_layers = 1
num_keypoints = 14
num_dim = 3

# %% functions
def create_samples_fast(dataset_sparse, dataset_dense, back_step = 1):
    dataXY = np.lib.stride_tricks.sliding_window_view(dataset_sparse, back_step, axis=0) # (n_samples/batch, n_features, back_step/seq)
    dataXY_dense = np.lib.stride_tricks.sliding_window_view(dataset_dense, back_step, axis=0) # (n_samples, n_features, back_step)
    dataX = np.moveaxis(dataXY, [0,1,2], [0,2,1]) # (batch, seq, feature)
    dataY = dataXY_dense[:, :, -1]   # (batch, feature)
    return dataX, dataY

def root_mean_squared_error(y_true, y_pred, pos): 
    return np.sqrt(np.mean(np.square(y_true[pos] - y_pred[pos])))

def root_mean_squared_error3d(y_true, y_pred, pos): 
    diff = y_true[pos] - y_pred[pos]
    diff3d = diff.reshape(-1, 3)
    return np.mean(np.sqrt(np.sum(np.square(diff3d), axis=1)))

def root_squared_error3d(y_true, y_pred, pos): 
    diff = y_true[pos] - y_pred[pos]
    diff3d = diff.reshape(-1, 3)
    return np.sqrt(np.sum(np.square(diff3d), axis=1))

np.random.seed(0)
torch.manual_seed(0)

# %% models
class CustomDataset(Dataset):
    def __init__(self, X, Y, augment = False):
        self.X = X   #(batch, seq, feature)
        self.Y_sparse = self.X[:,-1,:]  #(batch, feature) sparse
        self.Y = Y   #(batch, feature) dense
        self.augment = augment

    def __len__(self):
        return len(self.Y)

    def aug(self, X, Y_sparse, Y):
        look_back, nfeature = X.shape
        assert nfeature == Y_sparse.shape[0] == Y.shape[0]
        com_XY = torch.vstack((X, Y_sparse, Y))
        com_3d = com_XY.reshape(-1, nfeature//3, 3)  #(n_sample, n_bodyparts, 3)
        rad = torch.rand(1)*2*3.142   # 0-360
        ox, oy = com_3d[:,:,0], com_3d[:,:,1] #original dataset
        px, py = torch.randn(2)*2    # center point

        torch_com_3d = com_3d.clone()
        torch_com_3d[:,:,0] = ox + torch.cos(rad) * (px - ox) - torch.sin(rad) * (py - oy)
        torch_com_3d[:,:,1] = oy + torch.sin(rad) * (px - ox) + torch.cos(rad) * (py - oy)
        torch_com_XY = torch_com_3d.reshape(-1, nfeature)
        torch_X = torch_com_XY[:-2,:]
        torch_Y_sparse = torch_com_XY[-2,:]
        torch_Y = torch_com_XY[-1,:]

        return torch_X, torch_Y_sparse, torch_Y


    def __getitem__(self, idx):
        X, Y_sparse, Y = self.X[idx,:,:], self.Y_sparse[idx,:], self.Y[idx,:]
        if self.augment:
            X, Y_sparse, Y = self.aug(X, Y_sparse, Y)

        X[torch.isnan(X)] = missing_replacevalue
        Y_sparse[torch.isnan(Y_sparse)] = missing_replacevalue
        Y[torch.isnan(Y)] = missing_replacevalue
        return X, Y_sparse, Y  #(seq, feature), (feature, ), (feature, )

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Model, self).__init__()
        self.linerinput = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.liner = nn.Linear(hidden_size, output_size)
        self.linertrans = nn.Linear(hidden_size, 3)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        batchsize = x.shape[0]
        hidden = (torch.randn(self.num_layers, batchsize, self.hidden_size),
                  torch.randn(self.num_layers, batchsize, self.hidden_size))
        x = self.linerinput(x)
        x = self.relu(x)
        lstmout, hidden = self.lstm(x, hidden)
        lastout = lstmout[:,-1,:]
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
def create_datafromfile(rat_file):
    dense_tensor = sio.loadmat(rat_file)['points_3d']
    dim1, dim2, dim3 = dense_tensor.shape
    assert (dim2,dim3) == (num_keypoints, num_dim)
    random_tensor = np.random.rand(dim1, dim2)
    binary_tensor = random_tensor < missing_rate
    sparse_tensor = dense_tensor.copy()
    sparse_tensor[binary_tensor,:] = np.nan

    dense_mat = dense_tensor.reshape(dim1, dim2*dim3)   # (n_samples, n_features)
    sparse_mat = sparse_tensor.reshape(dim1, dim2*dim3)

    train_len = int((1 - test_rate) * dim1)
    train_sparse_mat = sparse_mat[:train_len, :]
    test_sparse_mat = sparse_mat[train_len - look_back+1:, :]
    train_dense_mat = dense_mat[:train_len, :]
    test_dense_mat = dense_mat[train_len - look_back+1:, :]

    # training_dataX, test_dataX: (batch, seq, feature)
    # training_dataY, test_dataY: (batch, feature)
    train_dataX, train_dataY = create_samples_fast(train_sparse_mat, train_dense_mat, look_back)
    test_dataX, test_dataY = create_samples_fast(test_sparse_mat, test_dense_mat, look_back)

    train_dataset = CustomDataset(torch.as_tensor(train_dataX, dtype = torch.float32),
                                torch.as_tensor(train_dataY, dtype = torch.float32),
                                augment = True)
    test_dataset = CustomDataset(torch.as_tensor(test_dataX, dtype = torch.float32),
                                torch.as_tensor(test_dataY, dtype = torch.float32))
    return train_dataset, test_dataset


def create_model(load_checkpoint):
    input_size = output_size = num_keypoints * num_dim
    model = Model(input_size, hidden_size, num_layers, output_size)
    if load_checkpoint:
        model.load_state_dict(torch.load(load_checkpoint))
    return model


def train_model(model, train_dataloader):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()
    n_epochs = 600

    for epoch in range(n_epochs):
        train_losses = []
        for X_batch, y_batch_sparse, y_batch in train_dataloader:        
            out = model(X_batch)   #(batch, feature)
            pos_nonzero = y_batch != missing_replacevalue
            optimizer.zero_grad()
            loss = mse_loss(out[pos_nonzero], y_batch[pos_nonzero])
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, np.mean(train_losses)))

    out_checkpoint = osp.join(osp.dirname(__file__), 'model.pth')
    torch.save(model.state_dict(), out_checkpoint)


def create_datasetloader(train_files):
    train_list, test_list = [], []
    for rat_file in train_files:
        train_dataset, test_dataset = create_datafromfile(rat_file)
        train_list.append(train_dataset)
        test_list.append(test_dataset)

    train_dataset = ConcatDataset(train_list)
    test_dataset =  ConcatDataset(test_list)

    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    return train_dataloader, test_dataloader


def test_model(model, test_dataloader):
    model.eval()
    with torch.no_grad():
        outpredlist = []
        outsparselist = []
        outdenselist = []
        for X_batch, y_batch_sparse, y_batch in test_dataloader:
            out= model(X_batch)
            outpredlist.append(out.detach().numpy())
            outsparselist.append(y_batch_sparse.detach().numpy())
            outdenselist.append(y_batch.detach().numpy())

    test_sparse_pred = np.concatenate(outpredlist, axis = 0)   # (batch, feature)
    test_sparse_ground_truth = np.concatenate(outsparselist, axis = 0)   # (batch, feature)
    test_dense_ground_truth  = np.concatenate(outdenselist, axis = 0)   # (batch, feature)
    return test_sparse_pred, test_sparse_ground_truth, test_dense_ground_truth


def pred_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        outpredlist = []
        for X_batch, y_batch_sparse, y_batch in dataloader:
            out= model(X_batch)
            outpredlist.append(out.detach().numpy())
    pred = np.concatenate(outpredlist, axis = 0)   # (batch, feature)
    return pred


def main(train_files):
    train_dataloader, test_dataloader = create_datasetloader(train_files)

    # %% create and train the model
    model = create_model(load_checkpoint)
    train_model(model, train_dataloader)

    # %% test the model
    test_sparse_pred, test_sparse_ground_truth, test_dense_ground_truth = test_model(model, test_dataloader)
    
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    print('Overall')
    pos = np.where(test_dense_ground_truth != missing_replacevalue)
    testPred_rmse = root_mean_squared_error(test_dense_ground_truth, test_sparse_pred, pos)
    testPred_cm = root_mean_squared_error3d(test_dense_ground_truth, test_sparse_pred, pos)
    print('Test prediction RMSE: %.2f RMSE, %.2f cm' % (testPred_rmse, testPred_cm))

    print('Imputation')
    posImpute = np.where((test_dense_ground_truth != missing_replacevalue) & (test_sparse_ground_truth == missing_replacevalue))
    testPred_rmse = root_mean_squared_error(test_dense_ground_truth, test_sparse_pred, posImpute)
    testPred_cm = root_mean_squared_error3d(test_dense_ground_truth, test_sparse_pred, posImpute)
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
        

    print('Self fitting')
    posFit = np.where((test_dense_ground_truth != missing_replacevalue) & (test_sparse_ground_truth != missing_replacevalue))
    testPred_rmse = root_mean_squared_error(test_dense_ground_truth, test_sparse_pred, posFit)
    testPred_cm = root_mean_squared_error3d(test_dense_ground_truth, test_sparse_pred, posFit)
    print('Test prediction RMSE: %.2f RMSE, %.2f cm' % (testPred_rmse, testPred_cm))

    sys.stdout = sys.__stdout__
    outstring = new_stdout.getvalue()
    print(outstring)
    return outstring


if __name__ == '__main__':
    main(train_files)
