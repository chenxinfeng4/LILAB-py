import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io as sio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

missing_replacevalue = -50

# %% Load data
# uniLSTM: create_samples_fast(dataset_sparse, dataset_dense, back_step = 23, forward_step = 0)
# biLSTM: create_samples_fast(dataset_sparse, dataset_dense, back_step = 12, forward_step = 12)
def create_randsparse(dense_tensor, missing_rate, n_side):
    [nseq, nbody, ndim] =  dense_tensor.shape
    nseqround = int((nseq // n_side) * n_side)
    nseqslice = nseqround // n_side
    random_tensor = np.zeros((nseq, nbody, ndim))
    random_tensor_slice = np.random.rand(nseqslice, nbody) < missing_rate
    for i in range(n_side):
        random_tensor[i:nseqround:n_side,:,:] = random_tensor_slice[:,:,np.newaxis]

    binary_tensor = random_tensor > 0
    return binary_tensor

def create_samples_fast(dataset_sparse, dataset_dense, back_step = 1, forward_step = 1):
    dataXY = np.lib.stride_tricks.sliding_window_view(dataset_sparse, back_step + forward_step + 1, axis=0) # (n_samples/batch, n_features, back_step/seq)
    dataXY_dense = np.lib.stride_tricks.sliding_window_view(dataset_dense, back_step+ forward_step + 1, axis=0) # (n_samples, n_features, back_step)
    dataX = np.moveaxis(dataXY, [0,1,2], [0,2,1]) # (batch, seq, feature)
    dataY = dataXY_dense[:, :, back_step]   # (batch, feature)
    dataX = np.ascontiguousarray(dataX)
    dataY = np.ascontiguousarray(dataY)
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

def create_randsparsefromdense(dense_tensor, missing_rate, n_side):
    binary_tensor = create_randsparse(dense_tensor, missing_rate, n_side)
    sparse_tensor = dense_tensor.copy()
    sparse_tensor[binary_tensor] = np.nan
    return sparse_tensor

# %% dataset
class CustomDataset(Dataset):
    def __init__(self, X, Y, look_back, augment = False):
        self.X = X   #(batch, seq, feature)
        self.Y_sparse = self.X[:,look_back,:]  #(batch, feature) sparse
        self.Y = Y   #(batch, feature) dense
        self.augment = augment

    def __len__(self):
        return len(self.Y)

    def aug(self, X, Y_sparse, Y):
        _, nfeature = X.shape
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


def create_datafromfile(rat_file, num_keypoints, num_dim, missing_rate,
                        test_rate, n_side, look_back, look_forward):
    dense_tensor = sio.loadmat(rat_file)['points_3d']
    dim1, dim2, dim3 = dense_tensor.shape
    assert (dim2,dim3) == (num_keypoints, num_dim)
    binary_tensor = create_randsparse(dense_tensor, missing_rate, n_side)
    sparse_tensor = dense_tensor.copy()
    sparse_tensor[binary_tensor] = np.nan

    dense_mat = dense_tensor.reshape(dim1, dim2*dim3)   # (n_samples, n_features)
    sparse_mat = sparse_tensor.reshape(dim1, dim2*dim3)

    train_len = int((1 - test_rate) * dim1)
    train_sparse_mat = sparse_mat[:train_len, :]
    test_sparse_mat = sparse_mat[train_len - look_back - look_forward+1:, :]
    train_dense_mat = dense_mat[:train_len, :]
    test_dense_mat = dense_mat[train_len - look_back - look_forward+1:, :]

    # training_dataX, test_dataX: (batch, seq, feature)
    # training_dataY, test_dataY: (batch, feature)
    train_dataX, train_dataY = create_samples_fast(train_sparse_mat, train_dense_mat, look_back, look_forward)
    test_dataX, test_dataY = create_samples_fast(test_sparse_mat, test_dense_mat, look_back, look_forward)

    train_dataset = CustomDataset(torch.as_tensor(train_dataX.copy(), dtype = torch.float32),
                                torch.as_tensor(train_dataY.copy(), dtype = torch.float32),
                                look_back, augment = True)
    test_dataset = CustomDataset(torch.as_tensor(test_dataX.copy(), dtype = torch.float32),
                                torch.as_tensor(test_dataY.copy(), dtype = torch.float32),
                                look_back, augment = False)
    return train_dataset, test_dataset


def create_datasetloader(train_files, batch_size, num_keypoints, num_dim, missing_rate,
                        test_rate, n_side, look_back, look_forward):
    train_list, test_list = [], []
    for rat_file in train_files:
        train_dataset, test_dataset = create_datafromfile(rat_file, num_keypoints, 
                        num_dim, missing_rate, test_rate, n_side, look_back, look_forward)
        train_list.append(train_dataset)
        test_list.append(test_dataset)

    train_dataset = ConcatDataset(train_list)
    test_dataset =  ConcatDataset(test_list)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader


def get_impute(test_sparse_mat):
    shapes = test_sparse_mat.shape
    test_sparse_mat = np.reshape(test_sparse_mat, (shapes[0], -1))
    df = pd.DataFrame(test_sparse_mat)
    df.interpolate(methods='linear', inplace=True)
    test_sparse_pred = df.values
    test_sparse_pred = np.reshape(test_sparse_pred, shapes)
    return test_sparse_pred

# %% models
def test_model(model, test_dataloader):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        outpredlist = []
        outsparselist = []
        outdenselist = []
        for X_batch, y_batch_sparse, y_batch in test_dataloader:
            X_batch, y_batch_sparse, y_batch = X_batch.to(device), y_batch_sparse.to(device), y_batch.to(device)
            out= model(X_batch)
            outpredlist.append(out.detach().cpu().numpy())
            outsparselist.append(y_batch_sparse.detach().cpu().numpy())
            outdenselist.append(y_batch.detach().cpu().numpy())

    test_sparse_pred = np.concatenate(outpredlist, axis = 0)   # (batch, feature)
    test_sparse_ground_truth = np.concatenate(outsparselist, axis = 0)   # (batch, feature)
    test_dense_ground_truth  = np.concatenate(outdenselist, axis = 0)   # (batch, feature)
    return test_sparse_pred, test_sparse_ground_truth, test_dense_ground_truth

def create_datasetloader_fromtensor(dense_tensor, num_keypoints, num_dim,
                                    look_back, look_forward):
    dim1, dim2, dim3 = dense_tensor.shape
    assert (dim2,dim3) == (num_keypoints, num_dim)
    
    dense_mat = dense_tensor.reshape(dim1, dim2*dim3)   # (n_samples, n_features)
    if look_forward == 0:
        densecopy_mat = np.vstack((dense_mat[:look_back,:], dense_mat))
    else:
        densecopy_mat = np.vstack((dense_mat[:look_back,:], dense_mat, dense_mat[-look_forward:,:]))

    dense_dataX, dense_dataY = create_samples_fast(densecopy_mat, densecopy_mat, look_back, look_forward)
    dense_dataX = np.ascontiguousarray(dense_dataX)
    dense_dataset = CustomDataset(torch.as_tensor(dense_dataX, dtype = torch.float32),
                                torch.as_tensor(dense_dataY, dtype = torch.float32),
                                look_back = look_back,
                                augment = False)
    dense_dataloader = DataLoader(dense_dataset, batch_size=200, shuffle=False)
    return dense_dataloader

def predict(model,dense_tensor, num_keypoints, num_dim,
            look_back, look_forward, 
            print_rmse = False):
    indnan = np.isnan(dense_tensor)
    dense_dataloader = create_datasetloader_fromtensor(dense_tensor, 
                                    num_keypoints, num_dim, look_back,
                                    look_forward)

    pred_mat, _, _ = test_model(model, dense_dataloader) # (batch, feature)

    dim1, dim2, dim3 = dense_tensor.shape
    dense_tensor_impute = pred_mat.reshape(dim1, dim2, dim3)
    dense_tensor_hybrid = dense_tensor.copy()
    dense_tensor_hybrid[indnan] = dense_tensor_impute[indnan]

    if print_rmse:
        print('Self fitting')
        posFit = np.where(~indnan)
        testPred_cm = root_mean_squared_error3d(dense_tensor, dense_tensor_impute, posFit)
        print('Test prediction error: %.2f cm' % (testPred_cm,))

    return dense_tensor_impute, dense_tensor_hybrid