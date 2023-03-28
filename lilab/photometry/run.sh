file="\\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\LS_NAC_fiberphotometry\PHO_04_NAC\0320行为记录1"

w2l "$file" | xargs python -m lilab.photometry.s1_tdms_2_photpkl
w2l "$file" | xargs python -m lilab.photometry.s2_photpkl_2_dFF
