cd /home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x9_mono_young
# ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-2*.mp4 \
# 1A. One batch
vfiles=$(ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/crop9/left/*.segpkl)
vfiles=$(ls /home/liying_lab/chenxinfeng/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/Shank3HETxWT_202210/2022-10-26*.segpkl)
\\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\multiview_9\chenxf\Shank3KOHet_2022-2023-d35
export volsize=170
echo "$vfiles" | sed 's/.segpkl/.mp4/' | cat -n |
    xargs -P 4 -l bash -c 'python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict-video-trt ../../configs/dannce_rat14_1280x800x9_max_config.yaml --vol-size $volsize --video-file $1 --gpu-id $(($0%4))'

# 1B. All batch
volsize_vfiles="
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-19_14-03-20Aw1xCb1.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-19_14-21-37Aw2xCb3.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-19_14-43-23Bw3xDb2.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-19_15-03-39Bw4xDb4.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-19_15-23-57Ew4xGb1.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-19_15-52-15Fw2xHb3.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-19_16-12-27Fw1xHb2.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-20_14-02-22Ew4xHb3.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-20_14-22-22Fw2xGb4.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-20_14-49-29Fw1xGb1.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-20_15-12-07Aw1xDb4.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-20_15-33-43Aw2xDb2.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-20_15-54-56Bw4xCb1.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-20_16-18-51Bw3xCb3.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-21_14-00-45Aw1xGb1.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-21_14-20-55Aw2xGb4.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-21_14-41-27Bw3xHb2.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-21_15-02-56Bw4xHb3.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-21_15-23-20Ew4xCb3.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-21_15-46-50Fw1xDb2.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-21_16-07-47Fw2xDb4.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-22_14-00-17Ew4xDb4.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-22_14-20-01Fw1xCb1.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-22_14-44-43Fw2xCb3.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-22_15-04-44Aw1xHb2.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-22_15-23-30Aw2xHb3.segpkl
250 /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/WTxWT_230625/testbatch3/2023-07-22_15-43-10Bw3xGb4.segpkl"

vdir="
/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/Shank3KOHet_2022-2023-d35/2022-02-24_16-39-47_SHANK21_KOxKO.segpkl
/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/Shank3KOHet_2022-2023-d35/2022-02-24_17-11-12_SHANK21_HetxHet.segpkl
/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/Shank3KOHet_2022-2023-d35/2022-02-24_17-44-31_SHANK20_KOxKO.segpkl
/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/Shank3KOHet_2022-2023-d35/2022-02-24_18-15-41_SHANK20_HetxHet.segpkl
/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/Shank3KOHet_2022-2023-d35/2022-02-25_15-27-40_SHANK21_wKOxbHet.segpkl
/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/Shank3KOHet_2022-2023-d35/2022-02-25_16-01-41_SHANK21_wHetxbKO.segpkl
/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/Shank3KOHet_2022-2023-d35/2022-02-25_16-35-53_SHANK20_wKOxbHet.segpkl
/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/Shank3KOHet_2022-2023-d35/2022-02-25_17-08-41_SHANK20_wHetxbKO.segpkl
/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/Shank3KOHet_2022-2023-d35/2022-02-26_17-51-06_SHANK21_KOxKO.segpkl
/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/Shank3KOHet_2022-2023-d35/2022-02-26_18-25-10_SHANK21_HetxHet.segpkl
/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/Shank3KOHet_2022-2023-d35/2022-02-26_18-59-15_SHANK20_KOxKO.segpkl
/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/Shank3KOHet_2022-2023-d35/2022-02-26_19-31-21_SHANK20_HetxHet.segpkl"

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
vfiles=$(ls /home/liying_lab/chenxinfeng/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/Shank3HETxWT_202210/2022-10-10/*.segpkl)
echo "$vfiles" | sed 's/.segpkl/.matcalibpkl/' | xargs -l -P 6 -r python -m lilab.smoothnet.s1_matcalibpkl2smooth_foot_cxf
echo "$vfiles" | sed 's/.segpkl/.smoothed_foot_cxf.matcalibpkl/' | xargs -P 6 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d $0 --iview 1 --postfix smoothed_foot --maxlen 9000'
echo "$vfiles" | sed 's/.segpkl/.matcalibpkl/' | xargs -l -P 6 -r python -m lilab.smoothnet.s1_matcalibpkl2smooth_foot_dzy
# 4B optional
echo "$vfiles" | sed 's/.segpkl/.smoothed_foot.matcalibpkl/' | xargs -P 8 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d_2view $0 --postfix smoothed_foot'

# 5 Concat video
echo "$vfiles" | sed 's/.segpkl//' | xargs -l -r bash -c 'python -m lilab.cvutils.concat_videopro $0_1_sktdraw.mp4 $0_1_sktdraw_smoothed_foot.mp4'
