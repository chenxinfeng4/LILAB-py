# %% imports
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
test_bilstm = False

if test_bilstm:
    from lilab.smoothpoints.biLSTM_point3d_impute_train import (create_model,
            create_datasetloader, root_mean_squared_error, root_squared_error3d,
            root_mean_squared_error3d, test_model, missing_replacevalue)
    load_checkpoint = '/home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/smoothpoints/modelbilstm.pth'
else:   
    from lilab.smoothpoints.LSTM_point3d_impute_train import (create_model,
            create_datasetloader, root_mean_squared_error, root_squared_error3d,
            root_mean_squared_error3d, test_model, missing_replacevalue)
    load_checkpoint = '/home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/smoothpoints/model.pth'

train_files = ['/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_side6/rat/rat_points3d_cm.mat',
               '/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_side6/15-37-42/white3d/rat_points3d_cm.mat']

# %% load model and dataset
train_dataloader, test_dataloader = create_datasetloader(train_files)
model = create_model(load_checkpoint)
test_sparse_pred, test_sparse_ground_truth, test_dense_ground_truth = test_model(model, test_dataloader)

# %% print
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

print('Self fitting')
posFit = np.where((test_dense_ground_truth != missing_replacevalue) & (test_sparse_ground_truth != missing_replacevalue))
testPred_rmse = root_mean_squared_error(test_dense_ground_truth, test_sparse_pred, posFit)
testPred_cm = root_mean_squared_error3d(test_dense_ground_truth, test_sparse_pred, posFit)
print('Test prediction RMSE: %.2f RMSE, %.2f cm' % (testPred_rmse, testPred_cm))


testPred_each_cm = root_squared_error3d(test_dense_ground_truth, test_sparse_pred, posImpute)


plt.figure(figsize=(8,8))
plt.hist(testPred_each_cm, bins=50)
plt.plot([testPred_cm, testPred_cm], [0, 350], 'k--')
plt.xlim([-0.2, 8.5])
plt.text(testPred_cm, 110, '%.2f cm' % testPred_cm, ha='right', va='bottom', fontsize=14)
plt.xlabel('Error (cm)')
plt.ylabel('Frequence')
plt.title('Imputation Error')
