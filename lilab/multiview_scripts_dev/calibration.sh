#conda activate mmpose
# vfile=/mnt/liying.cibr.ac.cn_Data_Temp/ZJF_lab/ball2
# vfile=/mnt/ftp.rat/multiview_9/SHANK3HETxWT/ball/2022-10-26_13-40-12Lball
# vfile=/home/liying_lab/chenxinfeng/DATA/2022-10-13_14-42-58Sball
vfile=/home/liying_lab/chenxinfeng/ftp.image_ZYQ/multiview_large/ball_only/20221125/2022-11-24_19-11-47ball

python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_faster $vfile.mp4 --pannels 9

python -m lilab.multiview_scripts_dev.s2_matpkl2ballpkl $vfile.matpkl  --time  9 18 23 31 38

python -m lilab.multiview_scripts_dev.s3_ballpkl_fliter $vfile.ballpkl  #删除重复的噪点，如果有必要

python -m lilab.multiview_scripts_dev.s3_ballpkl2calibpkl $vfile.ballpkl 

python -m lilab.multiview_scripts_dev.s4_matpkl2matcalibpkl $vfile.matpkl $vfile.calibpkl

python -m lilab.multiview_scripts_dev.s5_show_calibpkl2video  $vfile.matcalibpkl

python -m lilab.dannce.s1_ball2mat $vfile.calibpkl
