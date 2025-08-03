# conda activate mmdet
import sys
import runpy
import os

# os.chdir('demo/markerless_mouse_1')
os.chdir('/home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x9_mono_young')

# args = 'python -m lilab.mmpose.s3_kptpkl_2_video /mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/clips/2022-04-28_16-14-56_wwt_bwt_noiso_00time_9.kptpkl'
# args = 'python -m lilab.mmpose.s2_mmpose_pkl2matpkl_rat '
#args = 'python -m lilab.mmpose.s2_mmpose_pkl2matpkl_rat /mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/clips/2022-04-25_16-18-25_bwt_wwt_02time.mp4'
#args = 'python -m lilab.cvutils_new.crop_image /mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/checkbox/outframes'
# args = 'python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict ../../configs/dannce_mouse_config.yaml'
# args = 'python ../../dannce/utils/makeStructuredDataNoMocap.py ./DANNCE/predict_results/save_data_AVG0.mat ../../configs/mouse22_skeleton.mat ./label3d_dannce.mat'
# args = 'python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-train ../../configs/dannce_mouse_config.yaml --epochs=3'
# args = 'python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-train ../../configs/dannce_rat14_config.yaml'
# args = 'python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict ../../configs/dannce_rat14_config.yaml'

# args = 'python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-train ../../configs/dannce_rat14_800x600x6_config.yaml'
# args = 'python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict ../../configs/dannce_rat14_800x600x6_max_config.yaml  --gpu-id 2'

# --vol-size-list 280 240
# args = 'python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-train ../../configs/dannce_rat14_1280x800x9_max_config.yaml --gpu-id 0'
# args = 'python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-train ../../configs/dannce_marmoset_1280x800x6_max_config.yaml --gpu-id 0,1,2,3'
#args = 'python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict-video-trt ../../configs/dannce_rat14_1280x800x9_max_config.yaml --vol-size 240 --video-file /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/early_data/20221124/solo/2022-11-24_17-11-46_baseline2_rat1.mp4 --gpu-id 2'
args = 'python -m dannce.cli_trt ../../configs/dannce_rat14_1280x800x9_max_config.yaml  --vol-size 230 --video-file /mnt/liying.cibr.ac.cn_usb3/HTY/LFP/2025-02-18.4FurTest/b/2025-02-18_10-35-03.mp4  --gpu-id 2'
#'/home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict-video-trt ../../configs/dannce_rat14_1280x800x9_max_config.yaml --vol-size $1 --video-file $2 --gpu-id $(($0%4))
# /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict-video-trt ../../configs/dannce_rat14_1280x800x9_max_config.yaml --vol-size $1 --video-file $2 --gpu-id $(($0%4))
# args = 'python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-train ../../configs/dannce_rat14_1280x800x10_max_config.yaml --gpu-id 2'
# args = 'python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-train ../../configs/dannce_rat14_800x600x6_max_config.yaml --gpu-id 3'

# args = 'python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-train ../../configs/dannce_rat14_800x600x6_max_config.yaml  --gpu-id 2'
# com
# args = 'python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/com-train ../../configs/com_mouse_config.yaml'

args = args.split()
if args[0] == 'python':
    """pop up the first in the args"""
    args.pop(0)

if args[0] == '-m':
    """pop up the first in the args"""
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path


sys.argv.extend(args[1:])

fun(args[0], run_name='__main__')
