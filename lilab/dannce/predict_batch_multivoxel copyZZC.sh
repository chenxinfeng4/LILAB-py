conda activate mmdet
#!/bin/bash
# cd /home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x9_mono_young
cd /home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x9_mono_young

#"210 240" "270"
volsize_vfiles="
280 280 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/CNTNAP2_KO/cntnap2pnd75room2/a/2023-12-21_15-42-11D1bC1w.segpkl
280 280 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/CNTNAP2_KO/cntnap2pnd75room2/a/2023-12-21_15-16-15C1bD1w.segpkl
280 280 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/CNTNAP2_KO/cntnap2pnd75room2/a/2023-12-21_14-32-25A1bB1w.segpkl
280 280 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/CNTNAP2_KO/cntnap2pnd75room2/a/2023-12-21_14-54-14B1bA1w.segpkl
280 280 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/CNTNAP2_KO/cntnap2pnd75room2/a/2023-12-20_16-45-22D1bB2w.segpkl
280 280 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/CNTNAP2_KO/cntnap2pnd75room2/a/2023-12-20_15-54-28C1bA1w.segpkl
280 280 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/CNTNAP2_KO/cntnap2pnd75room2/a/2023-12-20_15-05-35F1bE2w.segpkl
280 280 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/CNTNAP2_KO/cntnap2pnd75room2/a/2023-12-20_14-46-10E2bF1w.segpkl
280 280 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/CNTNAP2_KO/cntnap2pnd75room2/a/2023-12-20_15-32-08A1bC1w.segpkl
280 280 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/CNTNAP2_KO/cntnap2pnd75room2/a/2023-12-20_14-21-51B1bA2w.segpkl
280 280 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/CNTNAP2_KO/cntnap2pnd75room2/a/2023-12-20_14-01-39A2bB1w.segpkl
280 280 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/CNTNAP2_KO/cntnap2pnd75room2/a/2023-12-19_16-15-17B1bD1w.segpkl
280 280 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/CNTNAP2_KO/cntnap2pnd75room2/a/2023-12-19_16-35-21D1bB1w.segpkl
280 280 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/CNTNAP2_KO/cntnap2pnd75room2/a/2023-12-19_15-39-49C1bA2w.segpkl
280 280 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/CNTNAP2_KO/cntnap2pnd75room2/a/2023-12-19_14-08-40A1bB2w.segpkl
280 280 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/CNTNAP2_KO/cntnap2pnd75room2/a/2023-12-19_14-53-21F2bE1w.segpkl
280 280 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/CNTNAP2_KO/cntnap2pnd75room2/a/2023-12-19_15-17-13A2bC1w.segpkl
280 280 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zhongzhenchao/CNTNAP2_KO/cntnap2pnd75room2/a/2023-12-19_14-29-46E1bF2w.segpkl

"

volsize_vfiles=$(echo "$volsize_vfiles" | grep -v '^[[:space:]]*$') #echo "$volsize_vfiles"
vfiles=$(echo "$volsize_vfiles" | awk '{print $3}')                 #echo "$vfiles"

echo "$volsize_vfiles" | sed 's/.segpkl/.mp4/' | cat -n |
    xargs -P 4 -l bash -c '/home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict-video-trt ../../configs/dannce_rat14_1280x800x9_max_config.yaml --vol-size-list $1 $2 --video-file $3 --gpu-id $(($0%3))'
# xargs -P 2 表示使用2个GPU，配合 choosecuda 0,1,2,3 确认使用gpu数量
#生成 .matcalibpkl 的文件，到此中断，再跑后面的

# 2. Convert file, one hit
# echo "$vfiles" | sed 's/.segpkl/.matcalibpkl/' | xargs -P 6 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d $0 --iview 3'

# 4B. Smooth foot_w16, body_w64, one hit
echo "$vfiles" | sed 's/.segpkl/.matcalibpkl/' | xargs -l -P 6 -r python -m lilab.smoothnet.s1_matcalibpkl2smooth_foot_dzy
echo "$vfiles" | sed 's/.segpkl/.smoothed_foot.matcalibpkl/' | xargs -P 6 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d $0 --iview 3 --postfix smoothed_foot '

# 4B optional
# echo "$vfiles" | sed 's/.segpkl/.smoothed_foot.matcalibpkl/' | xargs -P 8 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d_2view $0 --postfix smoothed_foot'
