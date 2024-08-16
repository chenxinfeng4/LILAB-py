# %% imports
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
import umap
from sklearn.preprocessing import MinMaxScaler
import ffmpegcv
from lilab.mmlab_scripts.show_pkl_seg_video_fast import default_mask_colors
import cv2
import os.path as osp
import tqdm

dannce_pkl = "/home/liying_lab/chenxinfeng/DATA/dannce/data/bw_rat_800x600x6_2022-2-24_shank3_voxel_anno_dannce.pkl"
mat_pkl = "/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-26_17-51-06_SHANK21_KOxKO.matcalibpkl"
seg_pkl = "/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-26_17-51-06_SHANK21_KOxKO.segpkl"
pkldata = pickle.load(open(dannce_pkl, "rb"))
matpkldata = pickle.load(open(mat_pkl, "rb"))
segpkldata = pickle.load(open(seg_pkl, "rb"))
# %%
data_3D = pkldata["data_3D"]
pts3d = data_3D.reshape((data_3D.shape[0], -1, 3))  # (N, K, 3)

# %%
boneindex = [
    (0, 1),
    (0, 2),
    (2, 3),
    (3, 6),
    (3, 7),
    (3, 8),
    (6, 7),
    (8, 9),
    (4, 5),
    (0, 5),
    (10, 11),
    (12, 13),
    (5, 12),
    (6, 8),
    (10, 12),
    (10, 11),
    (12, 13),
    (8, 12),
]
boneindex = np.array(boneindex)

# %%
def get_bonelength(pts3d):
    bonelength = np.linalg.norm(
        pts3d[..., boneindex[:, 0], :] - pts3d[..., boneindex[:, 1], :], axis=-1
    )
    return bonelength


bonelength = get_bonelength(pts3d)
bone_mean = np.nanmean(bonelength, axis=0)

for i in range(bonelength.shape[1]):
    bonelength[np.isnan(bonelength[:, i]), i] = bone_mean[i]


# pca = PCA(n_components=3)
pca = umap.UMAP()
pca.fit(bonelength)

feat = pca.transform(bonelength)


def pca_transform(bonelength):
    if bonelength.ndim == 1:
        bonelength = bonelength[None, :]
    for i in range(bonelength.shape[1]):
        bonelength[np.isnan(bonelength[:, i]), i] = bone_mean[i]

    bonelength_pca = pca.transform(bonelength)
    return bonelength_pca


plt.scatter(feat[:, 0], feat[:, 1])
# plt.scatter(feat[:,0], feat[:,2])

# %%
kptba = matpkldata["keypoints_xyz_ba"]
index1, index2 = 3259, 0
kptnow = kptba[index1, index2, ...]


bonelengthnow = get_bonelength(kptnow)[None, :]

bonelength_pca_now = pca_transform(bonelengthnow)


plt.scatter(feat[:, 0], feat[:, 1])
plt.scatter(bonelength_pca_now[:, 0], bonelength_pca_now[:, 1])

# %%
plt.scatter(feat[:, 0], feat[:, 2])
plt.scatter(bonelength_pca_now[:, 0], bonelength_pca_now[:, 2])

# %% 按夹角
boneangleindex = [
    (0, 1, 3),
    (0, 3, 2),
    (3, 6, 4),
    (3, 4, 8),
    (4, 10, 5),
    (4, 5, 12),
    (6, 10, 4),
    (8, 4, 12),
]
boneangleindex = np.array(boneangleindex)


def get_bonearea(pts3d):
    bonearea = np.zeros((pts3d.shape[0], len(boneangleindex)))
    anglebetween = np.zeros((pts3d.shape[0], len(boneangleindex) // 2))
    for i in range(pts3d.shape[0]):
        pts3d_now = pts3d[i, ...]
        P1 = pts3d_now[boneangleindex[:, 0], :]
        P2 = pts3d_now[boneangleindex[:, 1], :]
        P3 = pts3d_now[boneangleindex[:, 2], :]
        fa_vector = np.cross(P2 - P1, P3 - P1)
        bonearea[i] = np.linalg.norm(fa_vector, axis=-1) / 2
        fa_vector_unit = fa_vector / np.linalg.norm(fa_vector, axis=-1, keepdims=True)
        fa_P1 = fa_vector_unit[::2, :]
        fa_P2 = fa_vector_unit[1::2, :]
        anglebetween[i] = np.arccos(np.clip(np.sum(fa_P1 * fa_P2, axis=-1), -1, 1))

    return bonearea, anglebetween


bonearea, anglebetween = get_bonearea(pts3d)


def concat(*args):
    return np.concatenate(args, axis=1)


def nan_to_mean(features):
    features_mean = np.nanmean(features, axis=0)
    for i in range(features.shape[1]):
        features[np.isnan(features[:, i]), i] = features_mean[i]
    return features


features = concat(bonearea, anglebetween)
features = nan_to_mean(features)
scaler = MinMaxScaler()
features_s = scaler.fit_transform(features)

pca = umap.UMAP()
features_t = pca.fit_transform(features_s)
plt.scatter(features_t[:, 0], features_t[:, 1])

kptba = matpkldata["keypoints_xyz_ba"]
index1, index2 = 4500, 0
kptnow = kptba[index1, index2, ...][None, :]

features_s1 = concat(*get_bonearea(kptnow))
features_s2 = scaler.transform(features_s1)
features_s3 = pca.transform(features_s2)

plt.scatter(features_t[:, 0], features_t[:, 1])
plt.scatter(features_s3[:, 0], features_s3[:, 1])

# %% multivariate normal distribution
feat_mean = np.nanmean(features_s, axis=0)
feat_cov = np.cov(features_s, rowvar=False)
from scipy.stats import multivariate_normal

var = multivariate_normal(feat_mean, feat_cov)

y = var.pdf(features_s)
y_log = np.log10(y + 1e-10)
plt.hist(y_log, bins=100)


kptba = matpkldata["keypoints_xyz_ba"]
index1, index2 = 1261, 0
kptnow = kptba[index1, index2, ...][None, :]

features_s1 = concat(*get_bonearea(kptnow))
features_s2 = scaler.transform(features_s1)
np.log10(var.pdf(features_s2) + 1e-10)


kptba = matpkldata["keypoints_xyz_ba"]
kptnow = kptba[:, 0, ...]

features_s1 = concat(*get_bonearea(kptnow))
features_s2 = scaler.transform(features_s1)
y_log_sample = np.log10(var.pdf(features_s2) + 1e-10)

kptnow_white = kptba[:, 1, ...]
features_s1 = concat(*get_bonearea(kptnow_white))
features_s2 = scaler.transform(features_s1)
y_log_sample_white = np.log10(var.pdf(features_s2) + 1e-10)

plt.plot(y_log_sample[:1000])
plt.plot(y_log_sample_white[:1000])
plt.xlim([500, 700])

# %%
vfile_in = "/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/2022-02-26_17-51-06_SHANK21_KOxKO_1_sktdraw.mp4"
vfile_out = osp.splitext(vfile_in)[0] + "_pval.mp4"
vid = ffmpegcv.VideoCaptureNV(vfile_in)
vid_out = ffmpegcv.VideoWriterNV(vfile_out, None, vid.fps)
min_iter = min([len(vid), len(y_log_sample_white)])

bgr_font_color = [[5, 0, 94], [128, 61, 0]]
for i in tqdm.tqdm(range(min_iter)):
    _, frame = vid.read()
    pvals = [y_log_sample[i], y_log_sample_white[i]]
    frame = cv2.putText(
        frame,
        "{:4.1f}".format(pvals[0]),
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        bgr_font_color[0],
        2,
    )
    frame = cv2.putText(
        frame,
        "{:4.1f}".format(pvals[1]),
        (20, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        bgr_font_color[1],
        2,
    )
    vid_out.write(frame)
vid_out.release()
vid.release()
