#conda activate mmpose
vfile=/mnt/liying.cibr.ac.cn_Data_Temp/ZJF_lab/ball2

python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl $vfile.mp4 --pannels 4 &\
echo "Finished S1" &\
python -m lilab.multiview_scripts_dev.s2_matpkl2ballpkl $vfile.matpkl  --time 0 4 7 10 13 &\
echo "Finished S2" &\
python -m lilab.multiview_scripts_dev.s3_ballpkl2calibpkl $vfile.ballpkl &\
echo "Finished S3" &\
python -m lilab.multiview_scripts_new.s4_matpkl2matcalibpkl $vfile.matpkl $vfile.calibpkl &\
echo "Finished S4" &\
python -m lilab.multiview_scripts_new.s5_show_calibpkl2video  $vfile.matcalibpkl &\
echo "Finished S5" &\
