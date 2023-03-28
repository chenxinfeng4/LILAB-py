# 1. 激活对应环境，输入相关地址配置
# conda activate mmpose
vfile=`w2l "\\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\LS_NAC_fiberphotometry\ball\校准2023-03-27_16-16-51.mp4"`
tball=" 0 8 17 25 38"

vfile=`echo "$vfile" | sed 's/.mp4//'`
# 2. 每个视角单独预测小球
python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_faster $vfile.mp4 --pannels 9

# 3. （可选）检查是否有坏点，修正
python -m lilab.multiview_scripts_dev.s5_show_calibpkl2video $vfile.matpkl
# python -m lilab.multiview_scripts_dev.s2_matpkl_filter $vfile.matpkl --iview 1 --iframe 1366 
# python -m lilab.multiview_scripts_dev.s2_matpkl_filter $vfile.matpkl --iview 5 --iframe 1475 
# python -m lilab.multiview_scripts_dev.s2_matpkl_filter $vfile.matpkl --iview 5 --iframe 2145 
# python -m lilab.multiview_scripts_dev.s2_matpkl_filter $vfile.matpkl --iview 5 --iframe 2203 
# python -m lilab.multiview_scripts_dev.s2_matpkl_filter $vfile.matpkl --iview 7 --iframe 2532 
# 4. 检测小球的5个位置定标
python -m lilab.multiview_scripts_dev.s2_matpkl2ballpkl $vfile.matpkl  --time  $tball

python -m lilab.multiview_scripts_dev.s3_ballpkl2calibpkl $vfile.ballpkl 

python -m lilab.multiview_scripts_dev.s4_matpkl2matcalibpkl $vfile.matpkl $vfile.calibpkl

python -m lilab.multiview_scripts_dev.s5_show_calibpkl2video $vfile.matcalibpkl

python -m lilab.dannce.s1_ball2mat $vfile.calibpkl


# 用已有的 calibpkl 验证新的小球视频，看看是否位置偏差
vfile=/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY50/ball/2023-02-05_15-06-36ball
vcalib=/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY50/ball/2023-01-14_15-38-38Mball
python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_faster $vfile.mp4 --pannels 9
python -m lilab.multiview_scripts_dev.s4_matpkl2matcalibpkl $vfile.matpkl $vcalib.calibpkl
python -m lilab.multiview_scripts_dev.s5_show_calibpkl2video  $vfile.matcalibpkl
