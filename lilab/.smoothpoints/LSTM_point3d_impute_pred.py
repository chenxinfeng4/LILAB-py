import numpy as np
import io
import sys
import torch
import scipy.io as sio
import os.path as osp
from torch.utils.data import DataLoader
from lilab.smoothpoints.LSTM_point3d_impute_train import (CustomDataset, root_mean_squared_error3d, 
                            create_model, num_keypoints, num_dim, look_back,
                            pred_model, create_samples_fast, root_mean_squared_error)

load_checkpoint = osp.join(osp.dirname(__file__), 'model.pth')
model = create_model(load_checkpoint)

rat_file = '/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_side6/15-37-42/white3d/rat_points3d_cm.mat'
mystdout = io.StringIO()

def create_datasetloader_fromtensor(dense_tensor):
    dim1, dim2, dim3 = dense_tensor.shape
    assert (dim2,dim3) == (num_keypoints, num_dim)
    
    dense_mat = dense_tensor.reshape(dim1, dim2*dim3)   # (n_samples, n_features)
    densecopy_mat = np.vstack((dense_mat[:look_back-1,:], dense_mat))

    dense_dataX, dense_dataY = create_samples_fast(densecopy_mat, densecopy_mat, look_back)
    dense_dataX = np.ascontiguousarray(dense_dataX)
    dense_dataset = CustomDataset(torch.as_tensor(dense_dataX, dtype = torch.float32),
                                torch.as_tensor(dense_dataY, dtype = torch.float32),
                                augment = False)
    dense_dataloader = DataLoader(dense_dataset, batch_size=200, shuffle=False)
    return dense_dataloader


def predict(dense_tensor, print_rmse = False):
    indnan = np.isnan(dense_tensor)
    dense_dataloader = create_datasetloader_fromtensor(dense_tensor)

    pred_mat = pred_model(model, dense_dataloader) # (batch, feature)

    dim1, dim2, dim3 = dense_tensor.shape
    dense_tensor_impute = pred_mat.reshape(dim1, dim2, dim3)
    dense_tensor_hybrid = dense_tensor.copy()
    dense_tensor_hybrid[indnan] = dense_tensor_impute[indnan]

    if print_rmse:
        mystdout.truncate(0)
        mystdout.seek(0)
        sys.stdout = mystdout
        print('Self fitting')
        posFit = np.where(~indnan)
        testPred_rmse = root_mean_squared_error(dense_tensor, dense_tensor_impute, posFit)
        testPred_cm = root_mean_squared_error3d(dense_tensor, dense_tensor_impute, posFit)
        print('Test prediction RMSE: %.2f RMSE, %.2f cm' % (testPred_rmse, testPred_cm))
        sys.stdout = sys.__stdout__
        print(mystdout.getvalue())

    return dense_tensor_impute, dense_tensor_hybrid


def main(rat_file):
    rat_points3d = sio.loadmat(rat_file)['points_3d']
    rat_points3d_impute, rat_points3d_hybrid = predict(rat_points3d, print_rmse = True)
    sio.savemat(osp.join(osp.dirname(rat_file), 'rat_points3d_cm_1impute.mat'), {'points_3d': rat_points3d_hybrid})
    return mystdout.getvalue()
    

if __name__ == '__main__':
    main(rat_file)
