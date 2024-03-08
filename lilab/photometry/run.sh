
# tdms
file=`w2l "\\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\LS_NAC_fiberphotometry\1stBatch_NAC\PHO_08_NAC\0326行为记录\LS08_001_20230326.tdms"`
echo "$file" | xargs python -m lilab.photometry.s1_tdms_2_photpkl
echo "$file" | sed s/tdms/photpkl/ | xargs python -m lilab.photometry.s2_photpkl_2_dFF

ibrain=1
ichan=2
echo "$file" | sed s/tdms/dFFphotpkl/ | xargs -I {} python -m lilab.photometry.s4_dFF_repick {} $ibrain $ichan


# matlab
source activate mmdet
file=`w2l "\\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\zyq_cpp_photometry"`
ibrain=0  #=0，单光纤。= 0 | 1, 双侧光纤
ichan=1   #0=405(运动)，1=470（绿色），2=565（红色）
echo "$file" | xargs python -m lilab.photometry.s1_fipgui_2_photpkl
echo "$file" | xargs python -m lilab.photometry.s2_photpkl_2_dFF
echo "$file" | xargs -I {} python -m lilab.photometry.s4_dFF_repick {} $ibrain $ichan


# matlab
file=`w2l "\\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\fibertest"`
ibrain=0
ichan=1
echo "$file" | xargs python -m lilab.photometry.s1_fipgui_2_photpkl
echo "$file" | xargs python -m lilab.photometry.s2_photpkl_2_dFF
echo "$file" | xargs python -m lilab.photometry.s2b_dFFpkl_2_matlab

