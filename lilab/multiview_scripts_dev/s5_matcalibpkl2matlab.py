# %%
import scipy.io as sio
import pickle
matcalibpkl = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-26_19-31-21_SHANK20_HetxHet.smoothed.matcalibpkl'

pkldata = pickle.load(open(matcalibpkl, 'rb'))
sio.savemat(matcalibpkl.replace('.matcalibpkl', '.mat'), pkldata)
