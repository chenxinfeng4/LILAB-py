vfile=`w2l "\\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\marmoset_camera3_cxf\2024-4-13-singlemarmoset\output_3.mp4"`
vfile=`w2l "\\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\marmoset_camera3_cxf\2024-2-21_marmosettracking\2024-02-21_15-07-36.mp4"`


python -m lilab.cvutils_new.extract_frames_pannel_crop_random $vfile  --npick 50 --setupname frank
