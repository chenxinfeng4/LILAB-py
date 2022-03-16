# %% imports
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import tqdm
from lilab.smoothpoints_test.missing_rate import detectTTL, line_shift
from lilab.smoothpoints.LSTM_base import predict, create_randsparsefromdense, get_impute
from lilab.smoothpoints.biLSTM_point3d_impute_train import (create_model,
        num_keypoints, num_dim, look_back, look_forward, hidden_size, n_side,
        missing_rate, n_side, root_squared_error3d)

load_checkpoint = '/home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/smoothpoints/checkpoint/' \
                  f'model_HD{hidden_size}_BK{look_back}_FW{look_forward}_SD{n_side}_final.pth'

test_files = ['/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_side6/rat/rat_points3d_cm.mat',
               '/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_side6/15-37-42/white3d/rat_points3d_cm.mat']

model = create_model(load_checkpoint)


# %%
pck_2cm_lstm = []
error_each_lstm = []
error_lstm = []
pck_2cm_inter = []
error_each_inter = []
error_inter = []
for test_file in tqdm.tqdm(test_files):
    dense_tensor = sio.loadmat(test_file)['points_3d']
    sparse_tensor = create_randsparsefromdense(dense_tensor, missing_rate, n_side)
    nan_tensor = np.ones_like(sparse_tensor)
    nan_tensor[~np.isnan(sparse_tensor)] = np.nan

    sparse_pred, sparse_hybrid = predict(model, sparse_tensor, num_keypoints, num_dim,
            look_back, look_forward, 
            print_rmse = False)
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

    # %% print
    print('\n-----------------------------------------------------')
    print('test file:', osp.basename(test_file))
    print('LSTM Imputation', end='\t')
    posImpute = np.where(np.isnan(sparse_tensor))
    testPred_cm = root_squared_error3d(dense_tensor, sparse_pred, posImpute)
    print('Test prediction RMSE: %.2f cm' % (np.nanmean(testPred_cm),))

    # LSTM + shift
    print('LSTM Shift Imputation', end='\t')
    posImpute = np.where(np.isnan(sparse_tensor))
    testPred_cm = root_squared_error3d(dense_tensor, sparse_predcopy, posImpute)
    pck_2cm_lstm.append(np.mean(testPred_cm<2))
    error_lstm.append(np.mean(testPred_cm))
    error_each_lstm.append(testPred_cm)
    print('Test prediction RMSE: %.2f cm' % (np.nanmean(testPred_cm),))

    # Interploate
    sparse_pred_inter = get_impute(sparse_tensor)
    print('Interploate Imputation', end='\t')
    posImpute = np.where(np.isnan(sparse_tensor) & ~np.isnan(sparse_pred_inter))
    testPred_cm = root_squared_error3d(dense_tensor, sparse_pred_inter, posImpute)
    pck_2cm_inter.append(np.mean(testPred_cm<2))
    error_inter.append(np.mean(testPred_cm))
    error_each_inter.append(testPred_cm)
    print('Test prediction RMSE: %.2f cm' % (np.nanmean(testPred_cm),))

pck_2cm_lstm = np.array(pck_2cm_lstm)
error_lstm = np.array(error_lstm)
pck_2cm_inter = np.array(pck_2cm_inter)
error_inter = np.array(error_inter)

# %%
error_each_inter_array = np.concatenate(error_each_inter, axis=0)
error_each_lstm_array = np.concatenate(error_each_lstm, axis=0)

def show_mean_std(data, title):
    print('\n-----------------------------------------------------')
    print(title)
    print('Mean±Std: %.2f ± %.2f' % (np.mean(data), np.std(data)))

error_each_inter_array = error_each_inter_array[~np.isnan(error_each_inter_array)]
error_each_lstm_array  = error_each_lstm_array[~np.isnan(error_each_lstm_array)]

print('\n-----------------------------------------------------')
print(f'Hidden_size{hidden_size} LookBack{look_back} LookForward{look_forward} NSide{n_side}')
show_mean_std(error_each_lstm_array, 'Error: LSTM Imputation')
show_mean_std(error_each_inter_array, 'Error: Interploate Imputation')

print('Percentile 95:LSTM Imputation ', np.percentile(error_each_lstm_array, 95).round(2))
print('Percentile 95:Interploate Imputation ', np.percentile(error_each_inter_array, 95).round(2))

exit()
# %% refine nan_tensor
# nan_tensor: visible point = nan, invisible point = 1
invis_tensor = np.array(nan_tensor==1, dtype=int)
diff_invis_tensor = np.diff(invis_tensor, axis=0)
ind_invis_tensor = np.where(diff_invis_tensor==1) 
invis_tensor[ind_invis_tensor] = 1
ind_invis_tensor = np.where(diff_invis_tensor==-1) 
a = ind_invis_tensor[0]
a += 2
invis_tensor[ind_invis_tensor] = 1

nan_tensor_refine = np.ones_like(sparse_tensor) * np.nan
nan_tensor_refine[invis_tensor==1] = 1

sparse_hybridcopy = np.copy(sparse_hybrid)
sparse_hybridcopy[nan_tensor==1] = sparse_predcopy[nan_tensor==1]

# %% plot
index_rg = (slice(0,4200), 9, 0)
xticks = np.arange(0, 4200) / 10 
plt.figure(figsize=(10,4))

plt.plot(xticks, (dense_tensor*nan_tensor)  [index_rg], color='k', linewidth= 2, label='Ground Truth')

# plt.plot(sparse_pred   [index_rg])
plt.plot(xticks, (sparse_hybridcopy)   [index_rg], color='#0072bd',  label='LSTM')
plt.plot(xticks, (sparse_pred_inter)   [index_rg], color='#d95319', label='Linear')
plt.plot(xticks, dense_tensor [index_rg], color='k')
plt.xlim(220, 250)
plt.ylim([-30, 30])
plt.legend()

plt.xlabel('Time (sec)')
plt.ylabel('Coordinate X (cm)')
plt.title('Imputation')

# plt.savefig("myImagePDF34-44.pdf", format="pdf", bbox_inches="tight")
plt.show()

# %%
