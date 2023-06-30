# %% imports
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from lilab.smoothpoints_test.missing_rate import detectTTL, line_shift
import pandas as pd

from lilab.smoothpoints.biLSTM_point3d_impute_train import (create_model,
        create_datasetloader, root_mean_squared_error, root_squared_error3d,
        root_mean_squared_error3d, test_model, predict, missing_replacevalue)

test_bilstm = False

train_files = ['/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_side6/rat/rat_points3d_cm.mat',
               '/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_side6/15-37-42/white3d/rat_points3d_cm.mat']

def get_impute(test_sparse_mat):
    shapes = test_sparse_mat.shape
    test_sparse_mat = np.reshape(test_sparse_mat, (shapes[0], -1))
    df = pd.DataFrame(test_sparse_mat)
    df.interpolate(methods='linear', inplace=True)
    test_sparse_pred = df.values
    test_sparse_pred = np.reshape(test_sparse_pred, shapes)
    return test_sparse_pred

# %%
test_file = train_files[0]
dense_tensor = sio.loadmat(test_file)['points_3d']
sparse_tensor = dense_tensor

nan_tensor = np.ones_like(sparse_tensor)
nan_tensor[~np.isnan(sparse_tensor)] = np.nan
sparse_pred, sparse_hybrid = predict(model, sparse_tensor, print_rmse=True)
sparse_nanpred = sparse_pred.copy()
sparse_nanpred[~np.isnan(sparse_tensor)] =  np.nan


sparse_predcopy = sparse_pred.copy()
sparse_predcopyreshape = sparse_predcopy.reshape(sparse_pred.shape[0], -1).T
sparse_tensorreshape = sparse_tensor.reshape(sparse_tensor.shape[0], -1).T

for sparse_pred_vector, sparse_tensor_vector in zip(sparse_predcopyreshape, sparse_tensorreshape):
    ttl = np.isnan(sparse_tensor_vector)
    ind_nan_ttl, dur_nan_ttl = detectTTL(ttl)
    ind_clean = np.logical_and(ind_nan_ttl>5, (ind_nan_ttl+dur_nan_ttl)<len(ttl)-5)
    ind_nan_ttl, dur_nan_ttl = ind_nan_ttl[ind_clean], dur_nan_ttl[ind_clean]
    for ind_nan_ttl_now, dur_nan_ttl_now in zip(ind_nan_ttl, dur_nan_ttl):
        ind_bg = ind_nan_ttl_now-1
        ind_ed = ind_nan_ttl_now+dur_nan_ttl_now
        XR1 = np.nanmean(sparse_pred_vector[ind_bg-1:ind_bg+1])
        XR2 = np.nanmean(sparse_pred_vector[ind_ed:ind_ed+2])
        YR1 = np.nanmean(sparse_tensor_vector[ind_bg-1:ind_bg+1])
        YR2 = np.nanmean(sparse_tensor_vector[ind_ed:ind_ed+2])
        XR = np.array([XR1, XR2])
        YR = np.array([YR1, YR2])
        XNow = sparse_pred_vector[ind_nan_ttl_now-1:ind_nan_ttl_now+dur_nan_ttl_now+1]
        YNow = line_shift(XR, YR, XNow)
        sparse_pred_vector[ind_nan_ttl_now-1:ind_nan_ttl_now+dur_nan_ttl_now+1] = YNow

# %%
index_rg = (slice(0,1800), 9, 0)
xticks = np.arange(0, 1800) / 10 - 60
plt.figure(figsize=(10,4))
plt.plot(xticks, sparse_tensor [index_rg], color='k', alpha=0.5)

#%%
plt.plot(xticks, (dense_tensor*nan_tensor)  [index_rg], color='k', linewidth= 2, label='Ground Truth')


# plt.plot(sparse_pred   [index_rg])
plt.plot(xticks, (sparse_predcopy*nan_tensor)   [index_rg], color='#0072bd',  label='LSTM')
plt.plot(xticks, (sparse_pred_inter*nan_tensor)   [index_rg], color='#d95319', label='Linear')
plt.xlim(34, 44)
plt.ylim([-25, 10])
# plt.legend()

plt.xlabel('Time (sec)')
plt.ylabel('Coordinate X (cm)')
plt.title('Imputation')

plt.savefig("myImagePDF34-44.pdf", format="pdf", bbox_inches="tight")
plt.show()
