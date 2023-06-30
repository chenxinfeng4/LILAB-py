import os.path as osp
import io
import sys
import scipy.io as sio
from lilab.smoothpoints.LSTM_base import predict
from lilab.smoothpoints.biLSTM_point3d_impute_train import (create_model,
        num_keypoints, num_dim, look_back, look_forward)


load_checkpoint = osp.join(osp.dirname(__file__), 'model_HD60_BK36_FW36_SD30_300.pth')
model = create_model(load_checkpoint)

rat_file = '/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_30fps/whiteblack/15-11-45-black/rat_points3d_cm.mat'
mystdout = io.StringIO()


def main(rat_file):
    mystdout.truncate(0)
    mystdout.seek(0)
    sys.stdout = mystdout
    rat_points3d = sio.loadmat(rat_file)['points_3d']
    rat_points3d_impute, rat_points3d_hybrid = predict(model, rat_points3d, num_keypoints, 
                                        num_dim, look_back, look_forward, print_rmse=True)
    sio.savemat(osp.join(osp.dirname(rat_file), 'rat_points3d_cm_1impute.mat'), {'points_3d': rat_points3d_hybrid})
    sys.stdout = sys.__stdout__
    return mystdout.getvalue()
    

if __name__ == '__main__':
    main(rat_file)
