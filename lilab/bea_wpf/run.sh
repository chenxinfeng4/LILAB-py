# 将单动物的3D关键点，导出到为 BeA_WPF 格式，然后可借助GUI软件分析
conda activate mmdet

vdir='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/zyq_to_dzy/total_20230411/multi'


python -m lilab.bea_wpf.s1_matcalibpkl_to_bea_3d $vdir --force-ratid b 

python -m lilab.bea_wpf.s2_create_bea_project $vdir/BeA_WPF