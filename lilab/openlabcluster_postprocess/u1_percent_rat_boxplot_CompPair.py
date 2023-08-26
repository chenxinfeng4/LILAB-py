#%%
#from openlabcluster.utils import auxiliaryfunctions
import os
import pickle as pkl
import numpy as np
import os.path as osp
import matplotlib
#matplotlib.use('agg') 
import matplotlib.pyplot as plt
from lilab.openlabcluster_postprocess.s1_merge_3_file import get_assert_1_file
import pandas as pd
from statannotations.Annotator import Annotator
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from numpy import linalg
#%%
##path
project='/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/SexAge/Day35'
bhv_seqFile = get_assert_1_file(osp.join(project,'*_svm2all*_sequences.pkl'))
bhvSeqs = pkl.load(open(bhv_seqFile,'rb'))
print(list(bhvSeqs.keys()))  #label: 0,1,2...,k_best

clippredpkl = get_assert_1_file(osp.join(project,'*.clippredpkl'))
clippredata = pkl.load(open(clippredpkl,'rb'))

k_best = clippredata['ncluster'] #label: 0,1,2...,k_best
assert len(clippredata['cluster_names']) == k_best  #不包含nonsocial
lab_names2 = [f'{lab} [{i:>2}]' for i, lab in enumerate(['Non social']+clippredata['cluster_names'])]
df_labnames = pd.DataFrame({'behlabel':range(k_best+1), 'lab_names':lab_names2})
df_labreorder = pd.DataFrame({'behlabel':range(k_best+1)})
# df_labnames = pd.merge(df_labnames, df_labreorder, on='behlabel')
# df_labnames = df_labnames.reindex([0]+clippredata['cluster_labels_ordered'].tolist()).reset_index(drop=True)
#%%
##read group information
groupFile=get_assert_1_file(osp.join(project,'rat_info/*rat_*info*.xlsx'))
rat_info=pd.read_excel(groupFile,sheet_name='rat_info',engine='openpyxl')
video_info=pd.read_excel(groupFile,sheet_name='熊组合作D35_treat_info',engine='openpyxl')

rat_info = rat_info.filter(regex=r'^((?!Unnamed).)*$')
video_info = video_info.filter(regex=r'^((?!Unnamed).)*$')

#%% 表的检查
assert {'animal', 'color', 'gender', 'dob'} <= set(rat_info.columns)
assert {'video_nake', 'animal', 'partner', 'usv_file'} <= set(video_info.columns)
df_merge_b = pd.merge(rat_info, video_info['animal'], on='animal', how='right')
assert (df_merge_b['color'] == 'b').all()
df_merge_w = pd.merge(rat_info, video_info['partner'], left_on='animal', right_on='partner', how='right')
assert (df_merge_w['color'] == 'w').all()
rats_black = rat_info[rat_info['color'] == 'b']['animal'].values
rats_male  = rat_info[rat_info['gender'] == 'male']['animal'].values

# %%
rows_list = []
for i in range(video_info.shape[0]):
    record_now = video_info.iloc[i]
    video_nake = record_now['video_nake']

    animal_ratid, partner_ratid = record_now['animal'], record_now['partner']
    is_blackfirst = animal_ratid in rats_black
    black_ratid, white_ratid = (animal_ratid, partner_ratid) if is_blackfirst else (partner_ratid, animal_ratid) #first_ratid is black_rat
    groupdict = {True:'male', False:'female'}
    group_name = groupdict[black_ratid in rats_male] + groupdict[white_ratid in rats_male]
    rows_list.append([black_ratid, white_ratid, group_name, video_nake, True])

    group_name = groupdict[white_ratid in rats_male] + groupdict[black_ratid in rats_male]
    rows_list.append([white_ratid, black_ratid, group_name, video_nake, False])

df_group = pd.DataFrame(columns=['first_ratid', 'partner_ratid', 'group', 'video_nake', 'is_blackfirst'],
                        data=rows_list)

df_group['beh_key'] = pd.Series(['fps30_' + v.replace('-','_') + '_startFrame0_' + ('blackFirst' if is_blackfirst else 'whiteFirst')
                        for v, is_blackfirst in zip(df_group['video_nake'], df_group['is_blackfirst'])])

# 对 Category 列进行分组并计数
count = df_group.groupby('group').size()

print(count)

# %%
def get_behavior_percentages(bhv):
    return [np.mean(bhv==c) for c in range(k_best+1)] 

freq_data = [get_behavior_percentages(np.array(bhvSeqs[beh_key])) for beh_key in df_group['beh_key']]
freq_data = np.array(freq_data)
#freq_data[:,1:] /= (1-freq_data[:,0])[:, None]
df_freq_data_list = []
for i, freq_data_this in enumerate(freq_data):
    df_freq_data = pd.DataFrame({'behlabel':range(k_best+1), 'freq':freq_data_this})
    df_freq_data['beh_key'] = df_group['beh_key'][i]
    df_freq_data_list.append(df_freq_data)

df_freq_data_list = pd.concat(df_freq_data_list, axis=0)

df_group_x_freq = pd.merge(pd.merge(df_group, df_freq_data_list, on='beh_key'), df_labnames, on='behlabel')
df_group_x_freq1 = df_group_x_freq[df_group_x_freq['behlabel']!=0]
##get groups information OK
# with pd.ExcelWriter(osp.join(project,'groups_combine.xlsx')) as writer:
#     df_group.to_excel(writer, sheet_name='info_all')


# plt.figure(figsize = (20,10))
# sns.boxplot(x='behlabel', y='freq', hue='group', 
#             hue_order=['malemale', 'femalefemale', 'malefemale', 'femalemale'], data=df_group_x_freq)

#%%
plt.figure(figsize = (10,15))
sns.boxplot(y='lab_names', x='freq', hue='group', 
            hue_order=['malemale', 'femalefemale'],
            data=df_group_x_freq,
            width=0.45,
            order=df_labnames['lab_names'].values,
            orient='h')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Percentage', fontsize=20)
plt.ylabel('Label', fontsize=20)
leg = plt.legend(fontsize=14)
leg.get_texts()[0].set_text('M_M')
leg.get_texts()[1].set_text('F_F')
# plt.savefig(osp.join(project,'DAY55_boxplot_MM_vs_FF2.pdf'), bbox_inches='tight')
# plt.yticks(np.arange(1, k_best+1))
# plt.gca().set_yticklabels(lab_names2[1:])
pairs = [[(label, 'malemale'), (label, 'femalefemale')] for label in df_labnames['lab_names'].values]
annot = Annotator(plt.gca(), pairs, plot='barplot',
                  x='freq', y='lab_names', hue='group', 
                  hue_order=['malemale', 'femalefemale'], 
                  orient='h', hide_non_significant=True,
                  order=df_labnames['lab_names'].values,
                  data=df_group_x_freq)

annot.configure(test='t-test_ind')
annot.apply_test()
annot.annotate()
#%%
for i in range(k_best+1):
    AB = df_group_x_freq[df_group_x_freq['behlabel']==i]
    A = AB[AB['group']=='malemale']
    B = AB[AB['group']=='femalefemale']
    t,p = stats.ttest_ind(A['freq'], B['freq'])
    if p<0.05:
       print('significnat', i)

#%%
plt.savefig(osp.join(project,'DAY55_boxplot_MM_vs_FF3.pdf'), bbox_inches='tight')

# %%
plt.figure(figsize = (10,15))
sns.boxplot(y='lab_names', x='freq', hue='group', 
            hue_order=['malefemale', 'femalemale'],
            data=df_group_x_freq,
            order=df_labnames['lab_names'].values,
            width=0.45,
            orient='h')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Percentage', fontsize=20)
plt.ylabel('Label', fontsize=20)
leg = plt.legend(fontsize=14)
leg.get_texts()[0].set_text('<M>_F')
leg.get_texts()[1].set_text('<F>_M')

pairs = [[(label, 'malefemale'), (label, 'femalemale')] for label in df_labnames['lab_names'].values]
annot = Annotator(plt.gca(), pairs, plot='barplot',
                  x='freq', y='lab_names', hue='group', 
                  hue_order=['malefemale', 'femalemale'], 
                  orient='h', hide_non_significant=True,
                  order=df_labnames['lab_names'].values,
                  data=df_group_x_freq)

annot.configure(test='t-test_ind')
annot.apply_test()
annot.annotate()

plt.savefig(osp.join(project,'DAY55_boxplot_MF_vs_FM2.pdf'), bbox_inches='tight')

#%%
for i in range(k_best+1):
    AB = df_group_x_freq[df_group_x_freq['behlabel']==i]
    A = AB[AB['group']=='malefemale']
    B = AB[AB['group']=='femalemale']
    t,p = stats.ttest_ind(A['freq'], B['freq'])
    if p<0.05:
       print('significnat', i)


#%%

pca = PCA(n_components=2)
data_pca = pca.fit_transform(freq_data)
data_dict = {'x': data_pca[:,0], 'y': data_pca[:,1], 'group': df_group['group'].values}
data_df = pd.DataFrame(data_dict)

def plt_ellipse(x, y, ax, color):
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                width=lambda_[0]*3, height=lambda_[1]*3,
                angle=-np.rad2deg(np.arccos(v[0, 0])), color=color)
    ell.set_facecolor('none')
    ax.add_artist(ell)

"""
def plt_ellipse(x, y, ax, color):
    data = np.array([x, y])
    pca_ = PCA(n_components=2)
    pca_.fit(data.T)
    A = pca_.get_covariance()

    center = np.array([np.mean(x), np.mean(y)])
    U, s, rotation = linalg.svd(A)
    rotation_matrix = np.dot(U, rotation)
    rotation_angle = np.arccos((np.trace(rotation_matrix) - 1) / 2)
    rotation_angle_degrees = np.degrees(rotation_angle)

    ell = Ellipse(xy=center,
                width=s[0]*3, height=s[1]*3,
                angle=rotation_angle_degrees, color=color)
    ell.set_facecolor('none')
    ax.add_artist(ell)
"""


plt.figure(figsize=(8,8))
list_geno = ['malemale', 'femalefemale', 'malefemale', 'femalemale']
legend_list_geno = ['M_M', 'F_F', '<M>_F', '<F>_M']
colors = ['#313695', '#a50026', '#74add1', '#f46d43']
ax = plt.gca()
for c, geno in zip(colors, list_geno):

    print(c, geno)
    data_geno = data_df[data_df['group']==geno]
    x, y = data_geno['x'], data_geno['y']
    plt_ellipse(x, y, ax, c)
    plt.scatter(x, y, c=c, s=60, label=geno)

plt.legend(list_geno)
plt.legend(legend_list_geno, fontsize=16)
plt.xticks([])
# plt.xlim([0.18, -0.18]) # swap the x axis
# plt.ylim([0.14, -0.12])
plt.yticks([-10])
plt.xlabel('PC-1', fontsize=20)
plt.ylabel('PC-2', fontsize=20)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


#%%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X_lda = PCA(n_components=8).fit_transform(freq_data[:,1:])  #not include nonsocial
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_lda,  df_group['group'].values)
data_dict = {'x': data_pca[:,0], 'y': data_pca[:,1], 'group': df_group['group'].values}
data_df = pd.DataFrame(data_dict)
data_df['LDA_1'] = X_lda[:, 0]
data_df['LDA_2'] = X_lda[:, 1]

plt.figure(figsize=(8,8))
list_geno = ['malemale', 'femalefemale', 'malefemale', 'femalemale']
legend_list_geno = ['M_M', 'F_F', '<M>_F', '<F>_M']
colors = ['#313695', '#a50026', '#74add1', '#f46d43']
ax = plt.gca()
for c, geno in zip(colors, list_geno):

    print(c, geno)
    data_geno = data_df[data_df['group']==geno]
    x, y = data_geno['LDA_1'], data_geno['LDA_2']
    plt_ellipse(x, y, ax, c)
    plt.scatter(x, y, c=c, s=60, label=geno)

plt.legend(legend_list_geno, fontsize=16)
plt.xticks([])
plt.yticks([-10])
plt.xlabel('LD-1', fontsize=20)
plt.ylabel('LD-2', fontsize=20)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(osp.join(project,'LDA_split_groups.pdf'), bbox_inches='tight')

# %%
list_geno = ['malemale', 'femalefemale']
colors = ['#313695', '#a50026']
legend_list_geno = ['M_M', 'F_F']
freq_data_sub = freq_data[(df_group['group'].values==list_geno[0]) | (df_group['group'].values==list_geno[1])]
group_sub = df_group['group'][(df_group['group'].values==list_geno[0]) | (df_group['group'].values==list_geno[1])]
pca = PCA(n_components=2)
X_lda = pca.fit_transform(freq_data_sub)

data_df = pd.DataFrame({'x': X_lda[:,0], 'y': X_lda[:,1], 'group': group_sub.values})


plt.figure(figsize=(8,8))
ax = plt.gca()

for c, geno in zip(colors, list_geno):
    print(c, geno)
    data_geno = data_df[data_df['group']==geno]
    x, y = data_geno['x'], data_geno['y']
    plt_ellipse(x, y, ax, c)
    plt.scatter(x, y, c=c, s=60, label=geno)

plt.legend(legend_list_geno, fontsize=16)
# plt.legend(fontsize=14)
plt.xticks([])
# plt.xlim([0.18, -0.18]) # swap the x axis
# plt.ylim([0.14, -0.12])
plt.yticks([-10])
plt.xlabel('PC-1', fontsize=20)
plt.ylabel('PC-2', fontsize=20)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#%%

def plt_ellipsoid(x, y, z, ax, color):
    data = np.array([x, y, z])
    pca_ = PCA(n_components=3)
    pca_.fit(data.T)
    A = pca_.get_covariance()

    center = np.array([np.mean(x), np.mean(y), np.mean(z)])
    U, s, rotation = linalg.svd(A)
    radii = np.sqrt(s) * 2
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 60)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

    ax.plot_surface(x, y, z,  rstride=3, cstride=3,  color=color, linewidth=0.1, alpha=0.2, shade=True)


#%%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = PCA(n_components=10)
X_lda = lda.fit_transform(freq_data)
lda = LinearDiscriminantAnalysis(n_components=3)
X_lda = lda.fit_transform(X_lda,  df_group['group'].values)

data_dict = {'x': data_pca[:,0], 'y': data_pca[:,1], 'group': df_group['group'].values}
data_df = pd.DataFrame(data_dict)
data_df['LDA_1'] = X_lda[:, 0]
data_df['LDA_2'] = X_lda[:, 1]
data_df['LDA_3'] = X_lda[:, 2]
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
list_geno = ['malemale', 'femalefemale', 'malefemale', 'femalemale']
legend_list_geno = ['M_M', 'F_F', '<M>_F', '<F>_M']
colors = ['#313695', '#a50026', '#74add1', '#f46d43']
ax = plt.gca()
for c, geno, genolabel in zip(colors, list_geno, legend_list_geno):

    print(c, geno)
    data_geno = data_df[data_df['group']==geno]
    x, y, z = data_geno['LDA_1'].values, data_geno['LDA_2'].values, data_geno['LDA_3'].values
    plt_ellipsoid(x, y, z, ax, c)
    ax.scatter(x, y, z, c=c, s=60, label=genolabel)

# ax.set_legend(legend_list_geno, fontsize=16)
# plt.xticks([])
# plt.yticks([-10])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_xlabel('LD-1', fontsize=20)
ax.set_ylabel('LD-2', fontsize=20)
ax.set_zlabel('LD-3', fontsize=20)
ax.set_title('Sexual Day 55', fontsize=20)
plt.legend(loc="upper right")
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.view_init(azim=270, elev=90) 

#%%
lda = PCA(n_components=3)
X_lda = lda.fit_transform(freq_data)
data_dict = {'x': data_pca[:,0], 'y': data_pca[:,1], 'group': df_group['group'].values}
data_df = pd.DataFrame(data_dict)
data_df['LDA_1'] = X_lda[:, 0]
data_df['LDA_2'] = X_lda[:, 1]
data_df['LDA_3'] = X_lda[:, 2]
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
list_geno = ['malemale', 'femalefemale', 'malefemale', 'femalemale']
legend_list_geno = ['M_M', 'F_F', '<M>_F', '<F>_M']
colors = ['#313695', '#a50026', '#74add1', '#f46d43']
ax = plt.gca()
for c, geno, genolabel in zip(colors, list_geno, legend_list_geno):

    print(c, geno)
    data_geno = data_df[data_df['group']==geno]
    x, y, z = data_geno['LDA_1'].values, data_geno['LDA_2'].values, data_geno['LDA_3'].values
    plt_ellipsoid(x, y, z, ax, c)
    ax.scatter(x, y, z, c=c, s=60, label=genolabel)

# ax.set_legend(legend_list_geno, fontsize=16)
# plt.xticks([])
# plt.yticks([-10])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_xlabel('PC-1', fontsize=20)
ax.set_ylabel('PC-2', fontsize=20)
ax.set_zlabel('PC-3', fontsize=20)
ax.set_title('Sexual Day 55', fontsize=20)
plt.legend(loc="upper right")
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

return
#for j in range(k_best+1):bhv_names.extend(dataUse.shape[0]*['clu_'+str(j)])
for j in range(k_best+1):bhv_names.extend(dataUse0.shape[0]*[lab_names[j]+'_'+str(j)])  #lab_names[j]+'_'+str(j)
pcts=pd.DataFrame({'Group':groupNames,'Cluster':bhv_names,'Percentage':pctsVect})  ##

##calculate Fold change first
pairNames=['Rank12','Rank13','Rank14','Rank23','Rank24','Rank34']
mean0=np.mean(dataUse0[dataUse0['pair']==pairNames[0]].iloc[:,1:],axis=0)
mean1=np.mean(dataUse0[dataUse0['pair']==pairNames[1]].iloc[:,1:],axis=0)
mean2=np.mean(dataUse0[dataUse0['pair']==pairNames[2]].iloc[:,1:],axis=0)
mean3=np.mean(dataUse0[dataUse0['pair']==pairNames[3]].iloc[:,1:],axis=0)
mean4=np.mean(dataUse0[dataUse0['pair']==pairNames[4]].iloc[:,1:],axis=0)
mean5=np.mean(dataUse0[dataUse0['pair']==pairNames[5]].iloc[:,1:],axis=0)
##folder changess
fc=np.mean(np.stack([mean0/mean5,mean1/mean5,mean2/mean5,mean3/mean5,mean4/mean5]),axis=0)
fc_inds=np.argsort(fc)[::-1]

##
lab_namesFC=[lab_names2[i] for i in fc_inds]
dataUseFC=pd.concat([dataUse0['pair'],dataUse0[fc_inds]],axis=1)
##equal line
x1=np.argmin(abs(fc[fc_inds]-1))
##add Anova star second
#%%
##plot behavior percentage in boxplot and Anova analysis, order By P values
'''
from scipy import stats
tPs=[]
fc14=dataUseFC[dataUseFC['pair']==name1]
fc12=dataUseFC[dataUseFC['pair']==name2]
for j in dataUseFC.columns[1:]:
   t,p=stats.ttest_ind(fc14[j],fc12[j])
   tPs.append(p)
'''

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
anova_Ps=tPs=[]

for j in dataUseFC.columns[1:]:
   duj=dataUseFC[['pair',j]]
   duj=duj.rename(columns={j:'value'})
   model=ols('value~pair',duj).fit()
   tPs.append(anova_lm(model)['PR(>F)']['pair'])

'''
##order by P value
inds=np.argsort(tPs)
P_ordered=np.array(tPs)[inds]
lab_names3=[lab_names2[i] for i in inds]
dataUse1=pd.concat([dataUse0['geno_in_pair'],dataUse0[inds]],axis=1)
'''

lab_names3=lab_namesFC
dataUse1=dataUseFC
##stars
stars=[]
#for pi in P_ordered:
for pi in tPs:
   if pi<0.001:
      stars.append('***')
   elif pi<0.01:
      stars.append('**')
   elif pi<0.05:
      stars.append('*')
   else:
      stars.append('')

##xlabel color by significance
xcolors=[]
for i,ap in enumerate(tPs):
   if ap<0.05 and i<x1:
      xcolors.append('brown')
   elif ap<0.05 and i>x1:
      xcolors.append('blue')
   else:
      xcolors.append('black')



##Order by increase to decrease of significant labels
##reshape for boxplot
dataUse1=dataUseFC
pctsVect=np.array(dataUse1.iloc[:,1:]).flatten('F')  ##flatten by colomns (8 files) 
groupNames=list(dataUse1.iloc[:,0])*(k_best+1)  ##group
bhv_names=[]
#for j in range(k_best+1):bhv_names.extend(dataUse.shape[0]*['clu_'+str(j)])
for j in range(k_best+1):bhv_names.extend(dataUse1.shape[0]*[lab_names3[j]])  
pcts=pd.DataFrame({'Groups':groupNames,'Cluster':bhv_names,'Percentage':pctsVect})  ##

##plot boxplot
import seaborn as sns
plt.close('all')
from statannotations.Annotator import Annotator
df=pcts
plt.figure(figsize=(30,15))
# variant names
x = "Cluster";y = "Percentage";hue = "Groups";hue_order = pairNames
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams["axes.labelsize"] = 18
plt.tick_params(labelsize=10)##font size
ax=sns.boxplot(x=x, y=y,hue=hue, hue_order=hue_order, palette=['pink','yellow','green','cyan','purple','gray'],data=pcts)
##plot star
for ki in range(k_best+1):
   kj=np.max(dataUse1.iloc[:,ki+1])+np.std(dataUse1.iloc[:,ki+1])*0.1
   plt.text(ki, kj,stars[ki],ha="center", va="center", color="black", fontsize=15)
ax.vlines(x1, 0, 0.24, linestyles='dashed', colors='gold')
##
plt.tight_layout()
plt.yticks(fontsize=15)
plt.xticks(range(k_best+1), lab_names3, rotation=85,fontsize=15,color=xcolors)
for kbi in range(k_best+1):
   ax.get_xticklabels()[kbi].set_color(xcolors[kbi])
plt.ylabel("Percentage",fontsize=15)
plt.xlabel("Label",fontsize=15)
bar_test_fig=osp.join(resPath,'manualLabels_percentages_comparison_AllPairs_FCorder_Anova_male.pdf')
plt.savefig(bar_test_fig,bbox_inches='tight')
#plt.close('all')

