# %%
import pickle
import matplotlib.pyplot as plt
import numpy as np

pklfile = "/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/Shank3HetxWT_230405/2023-04-09_13-57-52AwxCb.smoothed_foot.matcalibpkl"

# %%
pkldata = pickle.load(open(pklfile, "rb"))

# %%
kpt_xyz = pkldata["keypoints_xyz_ba"]

ego_animal = kpt_xyz[:, 0, :, :2]
ego_kpt_ind = [4, 5]
ego_kpt = ego_animal[:, ego_kpt_ind, :]
ego_start_xy = ego_kpt[:, 0, :]
ego_end_xy = ego_kpt[:, -1, :]
ego_vec = ego_end_xy - ego_start_xy

kpt_xyz_decenter = kpt_xyz[..., :2] - ego_start_xy[:, None, None, :]

# algin the ego_vec to x-axis
# get the sin and cos of the angle between ego_vec and x-axis
theta = np.arctan2(ego_vec[:, 1], ego_vec[:, 0])
theta_clockwise = theta + np.pi / 2
rotation_matrix = np.array(
    [
        [np.cos(theta_clockwise), np.sin(theta_clockwise)],
        [-np.sin(theta_clockwise), np.cos(theta_clockwise)],
    ]
).transpose(2, 0, 1)

ego_vec_aligned = np.matmul(rotation_matrix, ego_vec[:, :, None])[:, :, 0]

kpt_xyz_decenter_aligned = np.matmul(
    rotation_matrix[:, None, None, :, :], kpt_xyz_decenter[..., None]
)[..., 0]

kpt_xyz_tail = kpt_xyz_decenter_aligned[:100, 1, 5, :]
kpt_xyz_head = kpt_xyz_decenter_aligned[:100, 1, 0, :]
kpt_xyz_leftpow = kpt_xyz_decenter_aligned[:100, 1, 7, :]
kpt_xyz_rightpow = kpt_xyz_decenter_aligned[:100, 1, 9, :]
plt.scatter(kpt_xyz_tail[:, 0], kpt_xyz_tail[:, 1])
plt.scatter(kpt_xyz_head[:, 0], kpt_xyz_head[:, 1], c="r")
plt.scatter(kpt_xyz_leftpow[:, 0], kpt_xyz_leftpow[:, 1], c="g")
plt.scatter(kpt_xyz_rightpow[:, 0], kpt_xyz_rightpow[:, 1], c="b")
plt.axis([-150, 150, -150, 150])
plt.xlabel("x")

kpt_another = kpt_xyz_decenter_aligned[:, 1, 4, :]

heatmap, xedges, yedges = np.histogram2d(kpt_another[:, 0], kpt_another[:, 1], bins=100)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.imshow(heatmap.T, extent=extent, origin="lower", cmap="jet")
plt.colorbar()
plt.show()


# %%
centor_animal = 1
other_animal = 0
ego_animal = kpt_xyz[:, centor_animal, :, :2]
ego_kpt_ind = [4, 5]
ego_kpt = ego_animal[:, ego_kpt_ind, :]
ego_start_xy = ego_kpt[:, 0, :]
ego_end_xy = ego_kpt[:, -1, :]
ego_vec = ego_end_xy - ego_start_xy

kpt_xyz_decenter = kpt_xyz[..., :2] - ego_start_xy[:, None, None, :]

# algin the ego_vec to x-axis
# get the sin and cos of the angle between ego_vec and x-axis
theta = np.arctan2(ego_vec[:, 1], ego_vec[:, 0])
theta_clockwise = theta + np.pi / 2
rotation_matrix = np.array(
    [
        [np.cos(theta_clockwise), np.sin(theta_clockwise)],
        [-np.sin(theta_clockwise), np.cos(theta_clockwise)],
    ]
).transpose(2, 0, 1)

ego_vec_aligned = np.matmul(rotation_matrix, ego_vec[:, :, None])[:, :, 0]

kpt_xyz_decenter_aligned = np.matmul(
    rotation_matrix[:, None, None, :, :], kpt_xyz_decenter[..., None]
)[..., 0]

# kpt_xyz_tail = kpt_xyz_decenter_aligned[:100, 1, 5, :]
# kpt_xyz_head = kpt_xyz_decenter_aligned[:100, 1, 0, :]
# kpt_xyz_leftpow = kpt_xyz_decenter_aligned[:100, 1, 7, :]
# kpt_xyz_rightpow = kpt_xyz_decenter_aligned[:100, 1, 9, :]
# plt.scatter(kpt_xyz_tail[:,0], kpt_xyz_tail[:,1])
# plt.scatter(kpt_xyz_head[:,0], kpt_xyz_head[:,1], c='r')
# plt.scatter(kpt_xyz_leftpow[:,0], kpt_xyz_leftpow[:,1], c='g')
# plt.scatter(kpt_xyz_rightpow[:,0], kpt_xyz_rightpow[:,1], c='b')
# plt.axis([-150,150,-150,150])
# plt.xlabel('x')


plt.figure(figsize=(15, 30))
plt.subplot(1, 2, 1)
kpt_another = kpt_xyz_decenter_aligned[:, other_animal, 0, :]
heatmap, xedges, yedges = np.histogram2d(kpt_another[:, 0], kpt_another[:, 1], bins=100)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.imshow(heatmap.T, extent=extent, origin="lower", cmap="jet")
plt.title("Partner Head")
plt.plot([0, 0], [-100, 100], "w--", linewidth=2)
plt.plot([-100, 100], [0, 0], "w--", linewidth=2)
plt.axis([-100, 100, -100, 100])

plt.subplot(1, 2, 2)
kpt_another = kpt_xyz_decenter_aligned[:, other_animal, 4, :]
heatmap, xedges, yedges = np.histogram2d(kpt_another[:, 0], kpt_another[:, 1], bins=100)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.imshow(heatmap.T, extent=extent, origin="lower", cmap="jet")
plt.axis([-100, 100, -100, 100])
plt.plot([0, 0], [-100, 100], "w--", linewidth=2)
plt.plot([-100, 100], [0, 0], "w--", linewidth=2)
plt.title("Partner Center")
plt.show()
# %%
