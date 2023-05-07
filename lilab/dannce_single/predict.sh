
## 1 必选项参数
vdir='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zyq_to_dzy/20221116/solo'  # 要分析的视频文件/文件夹
vdir='/mnt/liying.cibr.ac_cn_Data_Temp/multiview_9/zyq_to_dzy/20221123/solo'
vcalib=`ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/dzy_oyy_ball/checked/0328-0330/solo/*.calibpkl`    # 小球矫正的文件
volsize=210
##  volsize for rats
# |         | Male | Female |
# | ------- | ---- | ------ |
# | DAY35   | 160  | 160    |
# | DAY50   | 190  | 180    |
# | DAY75   | 220  | 200    |
# | DAY成年 | 220  | 210    |

## 2. 推荐（默认）的模型文件
config='/home/liying_lab/chenxinfeng/DATA/mmpose/res50_coco_com2d_512x320_oyy.py'           # com2d 的模型配置文件
dannce_project='/home/liying_lab/chenxinfeng/DATA/dannce/demo_single/rat14_1280x800x9_mono' # dannce 的模型路径


## 3. 文件名解析
vfiles_nake=`ls $vdir/*.mp4 | egrep -v "com3d|sktdraw|mask" | sed 's/.mp4//'`
vfiles=`echo "$vfiles_nake"| xargs -I {} echo {}.segpkl`
volsize_vfiles=`echo "$vfiles" | xargs -I {} echo $volsize {}`


## 4. predict com3d
python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl_full_faster $vdir --pannels 9 --config $config
echo "$vfiles_nake" | xargs -n 1 -P 0 -I {} python -m lilab.multiview_scripts_dev.s4_matpkl2matcalibpkl {}.matpkl $vcalib
echo "$vfiles_nake" | xargs -n 1 -P 0 -I {} python -m lilab.dannce_single.s1_matcalibpkl_com3d_to_segpkl {}.matcalibpkl
# ls $vdir/*.segpkl | sed 's/.segpkl/.mp4/' | xargs -n 1 -P 0 python -m lilab.mmdet_dev.s4_segpkl_com3d_to_video --vox_size $volsize --maxlen 9000


## 5. dannce predict
cd $dannce_project

echo "$volsize_vfiles" | sed 's/.segpkl/.mp4/' | cat -n |
    xargs -P 4 -l bash -c '/home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict-video-trt ../../configs/dannce_rat14_1280x800x9_max_config.yaml --vol-size $1 --video-file $2 --gpu-id $(($0%4))'

echo "$vfiles" | sed 's/.segpkl/_dannce_predict.pkl/' |
    xargs -P 0 -l -r python -m lilab.dannce.s4_videopredictpkl2matcalibpkl

# Smooth foot_w16, body_w64, one hit
echo "$vfiles" | sed 's/.segpkl/.matcalibpkl/' | xargs -l -P 6 -r python -m lilab.smoothnet.s1_matcalibpkl2smooth_foot


## 6. 绘制 video
# plot video
# echo "$vfiles" | sed 's/.segpkl/.matcalibpkl/' | 
#     xargs -P 6 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d $0 --iview 1 --maxlen 9000'

echo "$vfiles" | sed 's/.segpkl/.smoothed_foot.matcalibpkl/' | 
    xargs -P 6 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d $0 --iview 1 --postfix smoothed_foot --maxlen 9000'


## 7. Clean
echo "$vfiles_nake" | xargs -n 1 -I {} rm {}.matpkl {}_dannce_predict* 
