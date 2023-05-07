file=`w2l "\\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\LS_NAC_fiberphotometry\videomerge\2023-03-27_15-53-08.dFFphotpkl"`

ibrain=1
ichan=2

echo "$file" | xargs python -m lilab.photometry.s1_tdms_2_photpkl
echo "$file" | xargs python -m lilab.photometry.s2_photpkl_2_dFF
echo "$file" | xargs -I {} python -m lilab.photometry.s4_dFF_repick {} $ibrain $ichan
