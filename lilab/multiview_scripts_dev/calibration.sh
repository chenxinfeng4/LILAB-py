# 1. 激活对应环境，输入相关地址配置
# conda activate mmpose
vfile=`w2l "\\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\LS_NAC_fiberphotometry\1stBatch_NAC\ball\2023-03-17_17-16-45.mp4"`


setupname=ana
tball=" 3 10 24 36 50 "
vfile=`echo "$vfile" | sed 's/.mp4//'`
# 2. 每个视角单独预测小球
python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_faster_2 $vfile.mp4 --config /home/liying_lab/chenxinfeng/DATA/mmpose/res50_coco_ball_512x320_cam9.py

# 3. （可选）检查是否有坏点，修正
python -m lilab.multiview_scripts_dev.s5_show_calibpkl2video $vfile.matpkl
# python -m lilab.multiview_scripts_dev.s2_matpkl_filter $vfile.matpkl --iview 1 --iframe 1366 
# python -m lilab.multiview_scripts_dev.s2_matpkl_filter $vfile.matpkl --iview 5 --iframe 1475 
# python -m lilab.multiview_scripts_dev.s2_matpkl_filter $vfile.matpkl --iview 5 --iframe 2145 
# python -m lilab.multiview_scripts_dev.s2_matpkl_filter $vfile.matpkl --iview 5 --iframe 2203 
# python -m lilab.multiview_scripts_dev.s2_matpkl_filter $vfile.matpkl --iview 7 --iframe 2532 
# 4. 检测小球的5个位置定标
python -m lilab.multiview_scripts_dev.s2_matpkl2ballpkl $vfile.matpkl  --time $tball --force-setupname $setupname

python -m lilab.multiview_scripts_dev.s3_ballpkl2calibpkl $vfile.ballpkl 

python -m lilab.multiview_scripts_dev.s4_matpkl2matcalibpkl $vfile.matpkl $vfile.calibpkl

python -m lilab.multiview_scripts_dev.s5_show_calibpkl2video $vfile.matcalibpkl

python -m lilab.dannce.s1_ball2mat $vfile.calibpkl

# 用已有的 calibpkl 验证新的小球视频，看看是否位置偏差
vfile=/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/Shank3HetxWT_230405/ball/2023-04-05_13-18-24ball
vcalib=/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/Shank3HetxWT_230405/ball/2023-04-11_13-23-39ball
python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_faster $vfile.mp4 --pannels 9
python -m lilab.multiview_scripts_dev.s4_matpkl2matcalibpkl $vfile.matpkl $vcalib.calibpkl
python -m lilab.multiview_scripts_dev.s5_show_calibpkl2video  $vfile.matcalibpkl

