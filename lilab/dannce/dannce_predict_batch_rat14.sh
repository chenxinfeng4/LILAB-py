vcd /home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x9_mono_young
# ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-2*.mp4 \
vfiles=$(ls /home/liying_lab/chenxinfeng/ftp.image_ZYQ/multiview_large/animal_only/20221124/*.segpkl)
vfiles=$(ls /home/liying_lab/chenxinfeng/ftp.image_ZYQ/multiview_large/animal_only/20221126/*.segpkl)
vfiles=$(ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/VPAxWT/SOLO/*.segpkl)
vfiles=$(ls /mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/SHANK3HETxWT/2022-10-13/*.segpkl)

vfiles=/home/liying_lab/chenxinfeng/ftp.image_ZYQ/multiview_large/animal_only/20221124/2022-11-24_18-58-34_control_male_rat1.segpkl

export volsize=165
echo "$vfiles" | sed 's/.segpkl/.mp4/' | cat -n |
    xargs -P 4 -l bash -c 'python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict-video-trt ../../configs/dannce_rat14_1280x800x9_max_config.yaml --vol-size $volsize --video-file $1 --gpu-id $(($0%4))'

# 2. Convert file, one hit
echo "$vfiles" | sed 's/.segpkl/_dannce_predict.pkl/' |
    xargs -P 0 -l -r python -m lilab.dannce.s4_videopredictpkl2matcalibpkl

echo "$vfiles" | sed 's/.segpkl/_dannce_predict.voxpkl/' | xargs rm
echo "$vfiles" | sed 's/.segpkl/_dannce_predict.pkl/' | xargs rm
echo "$vfiles" | sed 's/.segpkl/_dannce_predict.mat/' | xargs rm

# 3. Video generate. copy multi
echo "$vfiles" | sed 's/.segpkl/.matcalibpkl/' |
    xargs -P 4 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d $0 --iview 1 --maxlen 9000'

# 4A. Smooth, one hit
# echo "$vfiles" | sed 's/.segpkl/.matcalibpkl/' | xargs -l -P 2 -r python -m lilab.smoothnet.s1_matcalibpkl2smooth_shorter
# echo "$vfiles" | sed 's/.segpkl/.smoothed_w16.matcalibpkl/' | xargs -P 4 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d $0 --iview 1 --postfix smoothed_w16 --maxlen 9000'

# 4B. Smooth foot_w16, body_w64, one hit
echo "$vfiles" | sed 's/.segpkl/.matcalibpkl/' | xargs -l -P 4 -r python -m lilab.smoothnet.s1_matcalibpkl2smooth_foot
echo "$vfiles" | sed 's/.segpkl/.smoothed_foot.matcalibpkl/' | xargs -P 6 -l -r bash -c 'python -m lilab.mmpose.s3_matcalibpkl_2_video2d $0 --iview 1 --postfix smoothed_foot --maxlen 9000'

# 5 Concat video
echo "$vfiles" | sed 's/.segpkl//' | xargs -l -r bash -c 'python -m lilab.cvutils.concat_videopro $0_1_sktdraw.mp4 $0_1_sktdraw_smoothed_foot.mp4'
