# %%
import numpy as np
import matplotlib.pyplot as plt
import glob
from lilab.usv.s1_usv_tsv2wav import read_tsv_file, read_tsv_file_durationcut

tsvfolder = "/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/VPAxWT/USV"
tsvfiles = glob.glob(tsvfolder + "/*.tsv")

start_lists = [read_tsv_file_durationcut(tsvfile)[0] for tsvfile in tsvfiles]

plt.figure()
for start_list in start_lists:
    plt.hist(start_list, bins=20, density=True, alpha=0.5)
plt.show()

# %% histogram of start 
edges = np.linspace(0, 900, 20)
denses = np.array([np.histogram(start_list, bins=edges, density=True)[0] for start_list in start_lists 
                   if len(start_list)>10])

denses = np.array([np.histogram(start_list, bins=edges, density=False)[0] for start_list in start_lists 
                   if len(start_list)>10])

dense_mean = denses.mean(axis=0)
dense_std = denses.std(axis=0)
plt.figure()
plt.errorbar(edges[:-1], dense_mean, yerr=dense_std)
plt.xlabel('Time (s)')
plt.ylabel('USV Density')
# plt.yticks([])
plt.show()

# %%
n_usv = np.array([len(start_list) for start_list in start_lists])
n_usv_pre = n_usv[:4]
n_usv_post = n_usv[4:]
plt.figure()
plt.bar([0, 1], [n_usv_pre.mean(), n_usv_post.mean()], yerr=[n_usv_pre.std(), n_usv_post.std()])
plt.xticks([0, 1], ['35+', '50+'])
plt.xlabel('Age (day)')
plt.ylabel('USV Count')
plt.show()
