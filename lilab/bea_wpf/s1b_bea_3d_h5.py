# imports
# %%
import numpy as np
import h5py

h5_file = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zzc_to_ouyang/20230218_oxtr-c/SOLO2/BeA_WPF_2/results/BeAOutputs/rec-1-ABC-20230219165121_results.h5'

# read the h5 file
hf = h5py.File(h5_file, 'r')
# %%
hf_3d = hf['3Dskeleton']
a=hf_3d['Bodyparts']
b=hf_3d['FPS']
c=hf_3d['data3D']


Bodyparts = np.array([b'nose', b'left_ear', b'right_ear', b'neck', b'left_front_limb',
       b'right_front_limb', b'left_hind_limb', b'right_hind_limb',
       b'left_front_claw', b'right_front_claw', b'left_hind_claw',
       b'right_hind_claw', b'back', b'root_tail', b'mid_tail',
       b'tip_tail'], dtype='|S16')

FPS = np.array(30, dtype="<i4")

data3D = np.zeros((18000,48), dtype='float64')


data_file =  h5py.File(h5_file.replace('.h5', '_modifed.h5'), "w")
h5_3Dskeleton = data_file.create_group('3Dskeleton')

h5_3Dskeleton.create_dataset("FPS", data=FPS)
h5_3Dskeleton.create_dataset("data3D", data=data3D)
h5_3Dskeleton.create_dataset("Bodyparts", data=Bodyparts)

