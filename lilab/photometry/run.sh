file=`w2l "\\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\fibertest\20230525-black.tdms"`

ibrain=0
ichan=1

echo "$file" | xargs python -m lilab.photometry.s1_tdms_2_photpkl
echo "$file" | xargs python -m lilab.photometry.s2_photpkl_2_dFF
echo "$file" | xargs -I {} python -m lilab.photometry.s4_dFF_repick {} $ibrain $ichan


# matlab
file=`w2l "\\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\fibertest"`
ibrain=0
ichan=1
echo "$file" | xargs python -m lilab.photometry.s1_fipgui_2_photpkl
echo "$file" | xargs python -m lilab.photometry.s2_photpkl_2_dFF
echo "$file" | xargs -I {} python -m lilab.photometry.s4_dFF_repick {} $ibrain $ichan