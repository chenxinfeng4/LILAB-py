cd /home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x9_mono_young
# ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-2*.mp4 \
# 1A. One batch
vfiles=$(ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/crop9/left/*.segpkl)
vfiles=$(ls /home/liying_lab/chenxinfeng/liying.cibr.ac.cn_Data_Temp/multiview_9/zzc_to_dzy/youxianwancheng/video/*.segpkl)

export volsize=170
echo "$vfiles" | sed 's/.segpkl/.mp4/' | cat -n |
    xargs -P 4 -l bash -c 'python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict-video-trt ../../configs/dannce_rat14_1280x800x9_max_config.yaml --vol-size $volsize --video-file $1 --gpu-id $(($0%4))'

# 1B. All batch
volsize_vfiles="
220 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-12_15-17-00AwVPAxBbVPA.segpkl
220 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-12_15-35-08AbVPAxBwVPA.segpkl
220 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-12_16-11-36CbWTxDwWT.segpkl
190 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-13_13-59-32EbVPAxGwWT.segpkl
190 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-13_14-17-37EwVPAxGbWT.segpkl
190 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-13_14-39-35FwVPAxHbWT.segpkl
190 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-13_14-57-32FbVPAxHwWT.segpkl
220 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-13_15-20-18AbVPAxCwWT.segpkl
220 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-13_15-39-56AwVPAxCbWT.segpkl
220 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-13_15-59-28BwVPAxDbWt.segpkl
220 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-13_16-18-52BbVPAxDwWT.segpkl
190 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-14_13-59-26EwVPA.segpkl
190 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-14_14-18-57FbVPA.segpkl
220 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-14_15-16-36AwVPA.segpkl
220 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-14_15-34-37BbVPA.segpkl
220 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-15_13-58-19AwVPAXBbVPA.segpkl
220 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-15_14-41-03CwWTxDbWT.segpkl
220 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-15_15-01-05CbWTxDwWT.segpkl
190 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-15_15-40-42EbVPAxFwVPA.segpkl
190 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-16_17-53-39EbVPAxFwWT_Eblink.segpkl
220 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-16_18-33-33AwVPAxDbWT.segpkl
220 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/DAY75/2023-02-16_19-12-14BwVPAxCbWT.segpkl"

volsize_vfiles=$(echo "$volsize_vfiles" | grep -v '^[[:space:]]*$') #echo "$volsize_vfiles"
vfiles=$(echo "$volsize_vfiles" | awk '{print $2}')                 #echo "$vfiles"

echo "$volsize_vfiles" | sed 's/.segpkl/.mp4/' | cat -n |
    xargs -P 4 -l bash -c '/home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict-video-trt ../../configs/dannce_rat14_1280x800x9_max_config.yaml --vol-size $1 --video-file $2 --gpu-id $(($0%4))'

# 2. Convert file, one hit
# sleep 40m
echo "$vfiles" | sed 's/.segpkl/_dannce_predict.pkl/' |
    xargs -P 0 -l -r python -m lilab.dannce.s4_videopredictpkl2matcalibpkl

echo "$vfiles" | sed 's/.segpkl/_dannce_predict.voxpkl/' | xargs rm
echo "$vfiles" | sed 's/.segpkl/_dannce_predict.pkl/' | xargs rm
echo "$vfiles" | sed 's/.segpkl/_dannce_predict.mat/' | xargs rm

# 3. Video generate. copy multi
echo "$vfiles" | sed 's/.segpkl/.matcalibpkl/' |
    xargs -P 4 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d $0 --iview 1 --maxlen 9000'

# echo "$vfiles" | sed 's/.segpkl/.matcalibpkl/' |
#     xargs -P 4 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d_2view $0 --maxlen 9000'

# 4A. Smooth, one hit
# echo "$vfiles" | sed 's/.segpkl/.matcalibpkl/' | xargs -l -P 2 -r python -m lilab.smoothnet.s1_matcalibpkl2smooth_shorter
# echo "$vfiles" | sed 's/.segpkl/.smoothed_w16.matcalibpkl/' | xargs -P 4 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d $0 --iview 1 --postfix smoothed_w16 --maxlen 9000'

# 4B. Smooth foot_w16, body_w64, one hit
echo "$vfiles" | sed 's/.segpkl/.matcalibpkl/' | xargs -l -P 6 -r python -m lilab.smoothnet.s1_matcalibpkl2smooth_foot
echo "$vfiles" | sed 's/.segpkl/.smoothed_foot.matcalibpkl/' | xargs -P 6 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d $0 --iview 1 --postfix smoothed_foot --maxlen 9000'

# 4B optional
echo "$vfiles" | sed 's/.segpkl/.smoothed_foot.matcalibpkl/' | xargs -P 8 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d_2view $0 --postfix smoothed_foot'

# 5 Concat video
echo "$vfiles" | sed 's/.segpkl//' | xargs -l -r bash -c 'python -m lilab.cvutils.concat_videopro $0_1_sktdraw.mp4 $0_1_sktdraw_smoothed_foot.mp4'
