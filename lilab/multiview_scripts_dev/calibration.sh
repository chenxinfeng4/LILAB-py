#conda activate mmpose
# vfile=/mnt/liying.cibr.ac.cn_Data_Temp/ZJF_lab/ball2
vfile=/home/liying_lab/chenxinfeng/DATA/test_ball/2022-10-10_16-30-22before_Lball
# vfile=/home/liying_lab/chenxinfeng/DATA/2022-10-13_14-42-58Sball

python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl $vfile.mp4 --pannels 9

python -m lilab.multiview_scripts_new.s2_matpkl2ballpkl $vfile.matpkl  --time  0 9 20 30 45

python -m lilab.multiview_scripts_dev.s3_ballpkl_fliter $vfile.ballpkl  #删除重复的噪点，如果有必要

python -m lilab.multiview_scripts_new.s3_ballpkl2calibpkl $vfile.ballpkl 

python -m lilab.multiview_scripts_new.s4_matpkl2matcalibpkl $vfile.matpkl $vfile.calibpkl

python -m lilab.multiview_scripts_new.s5_show_calibpkl2video  $vfile.matcalibpkl
