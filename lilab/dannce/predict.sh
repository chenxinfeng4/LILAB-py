cd demo/rat14_800x600x6_mono_shank3
vfile=/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-26_17-51-06_SHANK21_KOxKO
python /home/liying_lab/chenxinfeng/.conda/envs/mmdet/bin/dannce-predict-video ../../configs/dannce_rat14_800x600x6_max_config.yaml --video-file $vfile.mp4 --gpu-id 3
python -m lilab.dannce.s4_videopredictpkl2matcalibpkl ${vfile}_dannce_predict.pkl
python -m lilab.mmpose.s3_matcalibpkl_2_video2d ${vfile}.matcalibpkl --iview 1
