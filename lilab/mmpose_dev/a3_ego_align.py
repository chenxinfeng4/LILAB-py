# from lilab.mmpose_dev.a3_ego_align import KptEgoAligner
# %%
import numpy as np
import pickle
import os.path as osp
import argparse

matcalibpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/SHANK3HETxWT/voxelmat_D50/2022-10-26_14-04-49M_bwtxwhet.smoothed_foot.matcalibpkl'
index_start = 4 # Back
index_end = 5   # Tail


class KptEgoAligner:
    def __init__(self, ikpt_start=index_start, ikpt_end=index_end):
        self.ikpt_start = ikpt_start
        self.ikpt_end = ikpt_end
        self.d_theta = 0

    def fit(self, kpt_xyz_ianimal:np.ndarray):
        shape = kpt_xyz_ianimal.shape  # (*_, nkpt, xyz)
        assert shape[-1] in [2,3] and shape[-2] >= self.ikpt_end
        kpt_xy_startend = kpt_xyz_ianimal[...,[self.ikpt_start,self.ikpt_end],:2] #(*_, 2, xy)
        theta = np.arctan2(kpt_xy_startend[...,1,1] - kpt_xy_startend[...,0,1],
                            kpt_xy_startend[...,1,0] - kpt_xy_startend[...,0,0])
        self.theta = theta
        self.origin_xy = kpt_xyz_ianimal[...,[self.ikpt_start],:2]  #(*_, 1, xy)
        self.origin_len = np.linalg.norm(kpt_xy_startend[...,0,:] - kpt_xy_startend[...,1,:], axis=-1) #(*_)
        self.scale_fact = np.mean(self.origin_len) / self.origin_len
        return theta
    
    def transform_fix(self, kpt_xyz_ianimal:np.ndarray, meanscaled=False):
        assert self.theta.ndim == kpt_xyz_ianimal.ndim - 2
        shape = kpt_xyz_ianimal.shape  # (*_, nkpt, xyz)
        assert shape[-1] in [2,3] and shape[-2] >= self.ikpt_end

        kpt_xy_ianimal = kpt_xyz_ianimal[..., :2] #(*_, nkpt, xy)
        theta_clockwise = self.theta + np.pi/2 + self.d_theta
        rotation_matrix = np.array([[np.cos(theta_clockwise), np.sin(theta_clockwise)], 
                                    [-np.sin(theta_clockwise), np.cos(theta_clockwise)]])
        rotation_matrix_fine = np.zeros((*theta_clockwise.shape, 2, 2))
        rotation_matrix_fine[...,0,0] = rotation_matrix[0,0]
        rotation_matrix_fine[...,0,1] = rotation_matrix[0,1]
        rotation_matrix_fine[...,1,0] = rotation_matrix[1,0]
        rotation_matrix_fine[...,1,1] = rotation_matrix[1,1] #(*_, 2, 2)

        kpt_xy_ianimal_decenter = kpt_xy_ianimal - self.origin_xy #(*_, nkpt, 2)

        kpt_xy_ianimal_rot = np.matmul(rotation_matrix_fine[...,None,:,:],
                                    kpt_xy_ianimal_decenter[...,None])[...,0]
        if meanscaled:
            kpt_xy_ianimal_rot = kpt_xy_ianimal_rot * self.scale_fact[...,None,None]
        kpt_xyz_ianimal_rot = kpt_xyz_ianimal.copy()
        kpt_xyz_ianimal_rot[...,:2] = kpt_xy_ianimal_rot
        return kpt_xyz_ianimal_rot
    
    def transform(self, kpt_xyz_ianimal:np.ndarray):
        assert self.theta.ndim == kpt_xyz_ianimal.ndim - 2
        shape = kpt_xyz_ianimal.shape  # (*_, nkpt, xyz)
        assert shape[-1] in [2,3] and shape[-2] >= self.ikpt_end

        kpt_xy_ianimal = kpt_xyz_ianimal[..., :2] #(*_, nkpt, xy)
        theta_clockwise = self.theta + np.pi/2
        rotation_matrix = np.array([[np.cos(theta_clockwise), np.sin(theta_clockwise)], 
                                    [-np.sin(theta_clockwise), np.cos(theta_clockwise)]])
        rotation_matrix_fine = np.zeros((*theta_clockwise.shape, 2, 2))
        rotation_matrix_fine[...,0,0] = rotation_matrix[0,0]
        rotation_matrix_fine[...,0,1] = rotation_matrix[0,1]
        rotation_matrix_fine[...,1,0] = rotation_matrix[1,0]
        rotation_matrix_fine[...,1,1] = rotation_matrix[1,1] #(*_, 2, 2)

        kpt_xy_ianimal_decenter = kpt_xy_ianimal - kpt_xy_ianimal[...,[self.ikpt_start],:] #(*_, nkpt, 2)

        kpt_xy_ianimal_rot = np.matmul(rotation_matrix_fine[...,None,:,:],
                                    kpt_xy_ianimal_decenter[...,None])[...,0]
        kpt_xyz_ianimal_rot = kpt_xyz_ianimal.copy()
        kpt_xyz_ianimal_rot[...,:2] = kpt_xy_ianimal_rot
        return kpt_xyz_ianimal_rot
    
    def fit_transform(self, kpt_xyz_ianimal:np.ndarray):
        self.fit(kpt_xyz_ianimal)
        return self.transform(kpt_xyz_ianimal)


def main(matcalibpkl:str):
    from lilab.bea_wpf.a2_matcalibpkl_to_c3d import convert_kpt_3d_to_c3d
    pkldata = pickle.load(open(matcalibpkl, 'rb'))
    out_c3d_file = osp.join(osp.dirname(matcalibpkl), osp.basename(matcalibpkl).split('.')[0] + '_align.c3d')
    kpt_xyz = pkldata['keypoints_xyz_ba']

    kA = KptEgoAligner()
    kA.fit(kpt_xyz)
    kpt_xyz_align = kA.transform(kpt_xyz)

    kpt_xyz_align[:,0,...,0] -= 100

    convert_kpt_3d_to_c3d(kpt_xyz_align, out_c3d_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('matcalibpkl', type=str)
    args = parser.parse_args()
    main(args.matcalibpkl)
