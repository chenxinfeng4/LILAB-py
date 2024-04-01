#!/bin/bash
#/home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/multiview_scripts_dev/p_calibration.sh $vfile carl
#vfile='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/202401_natural_social_interaction_group2/ball/BALL____2024-01-22_14-34-01.mp4'
#/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/202401_natural_social_interaction_group2/0117_PHO_social/BALL____2024-01-17_14-19-23.mp4
#/home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/multiview_scripts_dev/p_calibration.sh $vfile bob
#/home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/multiview_scripts_dev/p_calibration.sh /mnt/liying.cibr.ac.cn_Data_Temp_ZZC/2310shank3MDMA/ball/2023-10-21_13-47-08ball_room1/2023-10-21_13-47-08ball_room1.mp4 ana
#/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/cpp_conditioning/cpp_conditioning_day4_1213/ball_room1/BALL______2023-12-13_14-29-18.mp4
#/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/cpp_conditioning/cpp_conditioning_day5_1214/ball_room1/BALL_______2023-12-14_14-14-01.mp4
#/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/cpp_conditioning/cpp_conditioning_day3_1212/ball_room1/ball_____2023-12-12_11-12-25.mp4
#/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/cpp_conditioning/cpp_conditioning_day2_1211/ball_room1/2023-12-11_17-24-53ball.mp4
#/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/cpp_conditioning/cpp_conditioning_day1_1210/ball_room1/BALL____2023-12-10_19-38-34.mp4
#/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/cpp_conditioning/cpp_conditioning_day1_1208/ball_room2/BALL__2023-12-08_14-36-37.mp4
#/mnt/liying.cibr.ac.cn_Data_Temp/LS_NAC_fiberphotometry/1stBatch_NAC/ball/校准2023-03-24_16-33-16.mp4
#/mnt/liying.cibr.ac.cn_Data_Temp/LS_NAC_fiberphotometry/1stBatch_NAC/ball/cleaned/2023-03-14_18-02-20ball.mp4
#/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/cpp_conditioning/1216/BALL___________2023-12-16_14-16-46.mp4
#/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/cpp_conditioning/20231222_room2/2023-12-22_14-30-39ballroom2.mp4
#/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/cpp_conditioning/20231224_room2/BALL__2023-12-24_18-14-38.mp4
#/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhangyuanqing/202401_natural_social_interaction_group2/ball/BALL_____2024-01-12_17-29-15.mp4
if [ $# -lt 2 ]; then
    echo "缺少参数，请输入两个参数. 第一个为视频路径，第二个为设备名"
    exit 1
fi

vfilefull=$1
setupname=$2  # ana bob carl, 3套多相机设备的标志


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
