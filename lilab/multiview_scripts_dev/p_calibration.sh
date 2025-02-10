#!/bin/bash
#/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/ball/20231116/2023-11-16_17-14-23ball.mp4
#/home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/multiview_scripts_dev/p_calibration.sh $vfile carl
#vfile='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/202401_natural_social_interaction_group2/ball/BALL____2024-01-22_14-34-01.mp4'
#/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/202401_natural_social_interaction_group2/0117_PHO_social/BALL____2024-01-17_14-19-23.mp4

if [ $# -lt 2 ]; then
    echo "缺少参数，请输入两个参数. 第一个为视频路径，第二个为设备名"
    exit 1
fi

vfilefull=$1
setupname=$2  # ana:room1, bob:room2 small, carl:room2 big, 3套多相机设备的标志


vfile=`echo "$vfilefull" | sed 's/.mp4//'`
vfile_checkboard=$vfile
tball=" 0 0 0 0 10 " # 没有用，只是占位

# 2A. 设置棋盘格，全局定标X、Y、Z轴
python -m lilab.multiview_scripts_dev.p1_checkboard_global $vfile.mp4 --setupname $setupname --board_size 11 8 --square_size 20 &
# 2B. 每个视角单独预测小球,绘制2D轨迹视频
python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_faster $vfile.mp4 --pannels $setupname  #<深度学习>

# python -m lilab.multiview_scripts_dev.s5_show_calibpkl2video $vfile.matpkl &


# 3. （可选）检查是否有坏点，修正 

# 4. 多相机 relative pose
python -m lilab.multiview_scripts_dev.s2_matpkl2ballpkl $vfile.matpkl  --time $tball --force-setupname $setupname

python -m lilab.multiview_scripts_dev.s3_ballpkl2calibpkl $vfile.ballpkl --skip-camera-intrinsic --skip-global 

# 5. 用全局定标对齐，得到多相机 global pose
python -m lilab.multiview_scripts_dev.p2_calibpkl_refine_byglobal $vfile.calibpkl $vfile_checkboard.globalrefpkl

# 6. 合成小球3D轨迹，并绘制3D轨迹视频
python -m lilab.multiview_scripts_dev.s4_matpkl2matcalibpkl $vfile.matpkl $vfile.recalibpkl

python -m lilab.multiview_scripts_dev.s5_show_calibpkl2video $vfile.matcalibpkl 

# 7. 确定无误后，替换原来的 calibpkl
mv $vfile.recalibpkl $vfile.calibpkl

# 8. 导出为 matlab 格式，用于 label3D
python -m lilab.dannce.s1_ball2mat $vfile.calibpkl
