# python -m lilab.openlabcluster_postprocess.s4_moseq_like_motif_plot xxx.clippredpkl A/B/C/
#%%
import pickle
import numpy as np
import os
import os.path as osp
import re
import matplotlib.pyplot as plt
from lilab.mmpose_dev.a3_ego_align import KptEgoAligner
from matplotlib.patches import Polygon
import tqdm
import argparse

#%%
clippredpklfile = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/00_BehaviorAnalysis-seq2seq/sexDay55/kmeans/FWPCA0.00_P100_en3_hid30_epoch777_svm2allAcc0.93_kmeansK2use-38_fromK1-20_K100.clippredpkl'
matcalib_dir = '/DATA/taoxianming/rat/data/chenxinfeng/sexDay55/smoothfoot_all'


linkbody = np.array([[0,1],[0,2],[1,3],[2,3],[3,6],[6,7],[3,4],[4,5], [6,10],[8,12],
                     [3,8], [8,9],[5,10],[10,11],[5,12],[12,13]])
trianglebody = np.array([[0,1,2],[1,2,3],[3,4,6],[3,4,8],
                        [4,5,10],[4,5,12],[4,6,10],[4,8,12]])


def plot_skeleton_aframe_2d(ax, point2d_aframe, name, zorderobj):
    # identitycolor = '#004e82' if name=='white' else '#61000f'
    identitycolor = '#373737' if name=='black' else '#61000f'
    markersize = 6
    plt.plot(*(point2d_aframe[[3,6,7,10,11,8,9,12,13,0,1,2,5],:].T), linestyle = 'None',  zorder = zorderobj(),
            marker='o', markeredgecolor='none', markerfacecolor=identitycolor, markersize=markersize)[0]
    plt.plot(*(point2d_aframe[[4],:].T), linestyle = 'None', marker='o',  zorder = zorderobj(),
             markeredgecolor='none', markerfacecolor=identitycolor, markersize=4)[0]
    lineP_from = point2d_aframe[linkbody[:,0],:]
    lineP_to = point2d_aframe[linkbody[:,1],:]
    triangle_P1 = point2d_aframe[trianglebody[:,0],:]
    triangle_P2 = point2d_aframe[trianglebody[:,1],:]
    triangle_P3 = point2d_aframe[trianglebody[:,2],:]
    for (x1,y1), (x2,y2) in zip(lineP_from, lineP_to):
        plt.plot([x1, x2], [y1, y2], color=identitycolor, zorder = zorderobj(),linewidth=1)

    for (x1,y1), (x2,y2), (x3,y3) in zip(triangle_P1, triangle_P2, triangle_P3):
        triangle_vertices = [(x1,y1), (x2,y2), (x3,y3)]
        triangle = Polygon(triangle_vertices, closed=True, facecolor=identitycolor, alpha=0.3,  zorder = zorderobj())
        ax.add_patch(triangle)
    plt.draw()

def get_zorder():
    zorder = 0
    def get():
        nonlocal zorder
        zorder+=1
        return zorder
    return get

def get_align_kpt(kpt_twin_xyz, rot_degree=0):
    kpt_twin_b = kpt_twin_xyz[:,0]
    kpt_twin_w = kpt_twin_xyz[:,1]
    aligner = KptEgoAligner()
    aligner.fit(kpt_twin_w[[-1]])

    kpt_twin_b_ego = aligner.transform(kpt_twin_b)
    kpt_twin_w_ego = aligner.transform(kpt_twin_w)

    kpt_twin_cent = kpt_twin_b[0,0,:]
    kpt_twin_b_ego = kpt_twin_b - kpt_twin_cent
    kpt_twin_w_ego = kpt_twin_w - kpt_twin_cent

    theta_clockwise = aligner.theta[0] + np.pi/2 + rot_degree/360*2*np.pi#

    rotation_matrix_fine = np.array([[np.cos(theta_clockwise), np.sin(theta_clockwise), 0], 
                                    [-np.sin(theta_clockwise), np.cos(theta_clockwise), 0],
                                    [0, 0, 1]])
    kpt_xyz_b_rot = np.matmul(rotation_matrix_fine[...,None,:,:],
                                        kpt_twin_b_ego[...,None])[...,0]
    kpt_xyz_w_rot = np.matmul(rotation_matrix_fine[...,None,:,:],
                                        kpt_twin_w_ego[...,None])[...,0]
    kpt_xyz_b_rot[...,1]
    kpt_xyz_w_rot[...,1]
    return kpt_xyz_b_rot, kpt_xyz_w_rot


#%%
def get_align_kpt_proj(kpt_twin_xyz, plan_theta45, rot_degree=0):
    kpt_twin_b = kpt_twin_xyz[:,0]
    kpt_twin_w = kpt_twin_xyz[:,1]
    aligner = KptEgoAligner()
    aligner.fit(kpt_twin_w[[-1]])

    kpt_twin_b_ego = aligner.transform(kpt_twin_b)
    kpt_twin_w_ego = aligner.transform(kpt_twin_w)

    def proj(kpt_twin_xyz):
        kpt_twin_xp2 = p3d_project(kpt_twin_xyz, plan_theta45)
        return kpt_twin_xp2

    kpt_twin_cent = kpt_twin_w[-1,0,:]
    kpt_twin_b_ego = kpt_twin_b - kpt_twin_cent
    kpt_twin_w_ego = kpt_twin_w - kpt_twin_cent

    theta_clockwise = aligner.theta[0] + rot_degree/360*2*np.pi#

    rotation_matrix_fine = np.array([[np.cos(theta_clockwise), np.sin(theta_clockwise), 0], 
                                    [-np.sin(theta_clockwise), np.cos(theta_clockwise), 0],
                                    [0, 0, 1]])
    kpt_xyz_b_rot_orig = np.matmul(rotation_matrix_fine[...,None,:,:],
                                        kpt_twin_b_ego[...,None])[...,0]
    kpt_xyz_w_rot_orig = np.matmul(rotation_matrix_fine[...,None,:,:],
                                        kpt_twin_w_ego[...,None])[...,0]
    kpt_xyz_bw_rot_orig = np.concatenate([kpt_xyz_b_rot_orig, kpt_xyz_w_rot_orig], axis=0)

    origin_pt = np.array([kpt_xyz_bw_rot_orig[...,0].min() - 10, kpt_xyz_bw_rot_orig[...,1].min() + 10, 0])[None]
    kpt_xyz_b_rot = proj(kpt_xyz_b_rot_orig)
    kpt_xyz_w_rot = proj(kpt_xyz_w_rot_orig)
    origin_pt_p = proj(origin_pt)

    return kpt_xyz_b_rot_orig, kpt_xyz_w_rot_orig, kpt_xyz_b_rot, kpt_xyz_w_rot, origin_pt_p


def draw_point(kpt_xy_b_rot, kpt_xy_w_rot):
    kpt_buget = np.array([kpt_xy_b_rot, kpt_xy_w_rot])
    kpt_xmin = np.nanmin(kpt_buget[...,0])
    kpt_xmax = np.nanmax(kpt_buget[...,0])
    kpt_ymin = np.nanmin(kpt_buget[...,1])
    kpt_ymax = np.nanmax(kpt_buget[...,1])

    itao =  0
    alpha = 0.4
    zorderobj = get_zorder()
    ax = plt.gca()

    for itao in range(len(kpt_xy_b_rot)):
        if itao>0:
            ax.fill_between([kpt_xmin-10,kpt_xmax+10], y1=kpt_ymin-10, y2=kpt_ymax+10, 
                        facecolor='w', alpha=alpha, clip_on=False, zorder=zorderobj())
        plot_skeleton_aframe_2d(ax, kpt_xy_b_rot[itao], 'white', zorderobj=zorderobj)
        plot_skeleton_aframe_2d(ax, kpt_xy_w_rot[itao], 'black', zorderobj=zorderobj)
    return zorderobj

def plot_y_z_axis(ax, origin, axistype, theta, length, zorderobj):
    from lilab.comm_signal.line_scale import line_scale
    if axistype=='x':
        theta_end = 0
    elif axistype=='y':
        theta_end = line_scale([0, np.pi/2], [-1/2*np.pi, -3/4*np.pi], theta)
    elif axistype=='z':
        theta_end = line_scale([0, np.pi/2], [3/4*np.pi, 1/2*np.pi], theta)
    else:
        raise NotImplementedError
    tgt = origin + length * np.array([np.cos(theta_end), np.sin(theta_end)])
    tgt_txt = np.squeeze(origin + length*1.4* np.array([np.cos(theta_end), np.sin(theta_end)]))
    if axistype=='x':
        tgt_txt += [-2, 4]
    elif axistype=='y':
        tgt_txt += [0, -4]
    ax.plot(*np.squeeze(np.stack([origin, tgt], axis=0).T), color='k', linewidth=2, zorder=zorderobj())
    ax.text(*tgt_txt, axistype, fontsize=14, zorder=zorderobj())

#%%
def plot_an_axes(ax:plt.Axes, plan_theta45, kpt_twin_xyz, rot_degree=0):
    kpt_xyz_b_rot_orig, kpt_xyz_w_rot_orig, kpt_xyz_b_rot, kpt_xyz_w_rot, origin_pt_p = get_align_kpt_proj(kpt_twin_xyz, plan_theta45, rot_degree=rot_degree)

    vertics_shadow_b = kpt_xyz_b_rot_orig[-1][[0,1,6,10,5,12,8,2], :2]
    vertics_shadow_w = kpt_xyz_w_rot_orig[-1][[0,1,6,10,5,12,8,2], :2]
    floor_z = np.array([kpt_xyz_b_rot_orig[-1,..., -1], kpt_xyz_w_rot_orig[-1,..., -1]]).min()
    vertics_shadow_b = np.concatenate([vertics_shadow_b, floor_z*np.ones_like(vertics_shadow_b[...,:1])], axis=-1)
    vertics_shadow_w = np.concatenate([vertics_shadow_w, floor_z*np.ones_like(vertics_shadow_w[...,:1])], axis=-1)

    alpha=0.6
    vertics_shadow_b_proj = p3d_project(vertics_shadow_b, plan_theta45)
    polygon_b = Polygon(vertics_shadow_b_proj, closed=True, facecolor='k', alpha=alpha)
    vertics_shadow_w_proj = p3d_project(vertics_shadow_w, plan_theta45)
    polygon_w = Polygon(vertics_shadow_w_proj, closed=True, facecolor='k', alpha=alpha)

    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_ylabel('(X-Y Plane)')

    ax.grid()
    ax.tick_params(length=0)
    kpt_xy_b_rot = kpt_xyz_b_rot[...,[0,1]]
    kpt_xy_w_rot = kpt_xyz_w_rot[...,[0,1]]
    ax.add_patch(polygon_b)
    ax.add_patch(polygon_w)
    zorderobj = draw_point(kpt_xy_b_rot, kpt_xy_w_rot)
    for axistype in ['x', 'y', 'z']:
        plot_y_z_axis(ax, origin_pt_p, axistype, plan_theta45, length=10, zorderobj=zorderobj)


def p3d_project(p3d, plan_theta):
    if p3d.shape[-1]==2:
        p3d = np.concatenate([p3d, np.zeros_like(p3d[...,:1])], axis=-1)
    p3d_yz = p3d[...,1:]
    p3d_theta = np.arctan2(p3d_yz[...,1], p3d_yz[...,0])
    p3d_len = np.linalg.norm(p3d_yz, axis=-1)
    p3d_theta_plan = np.pi - (p3d_theta + plan_theta)
    proj_p3d = p3d_len * np.cos(p3d_theta_plan)
    proj_p2d = np.stack([p3d[...,0], proj_p3d], axis=-1)
    return proj_p2d


def parsename(segmentname):
    imagebasename = osp.basename(segmentname)
    pattern = re.compile(r'(.+)\.smoothed_foot.matcalibpkl.*startFrameReal(\d+)')
    vname, frameid = pattern.findall(imagebasename)[0]
    frameid = int(frameid)
    return vname, frameid


def main(clippredpklfile, matcalib_dir):
    clipdata = pickle.load(open(clippredpklfile, 'rb'))
    assert {'ncluster', 'ntwin', 'cluster_labels', 'embedding', 'embedding_d2', 'clipNames'}
    cluster_list = np.arange(clipdata['ncluster'])+1
    embedding = clipdata['embedding']
    cluster_labels = clipdata['cluster_labels']
    clip_typical_list = []
    for icluster in cluster_list:
        cluster_mask = cluster_labels == icluster
        clipdata_mask = clipdata['clipNames'][cluster_mask]
        cluster_embedding = embedding[cluster_mask]

        cluster_embedding_mean = cluster_embedding.mean(axis=0, keepdims=True)
        cluster_embedding_dist = np.linalg.norm(cluster_embedding - cluster_embedding_mean, axis=1)
        ind_sort = np.argsort(cluster_embedding_dist)
        clipdata_mask_sort = clipdata_mask[ind_sort]
        clipdata_mask_sort_blackfirst = [f for f in clipdata_mask_sort if 'blackFirst' in f]
        cliptypical = clipdata_mask_sort_blackfirst[0]
        vname, frameid = parsename(cliptypical)
        clip_typical_list.append((icluster, vname, frameid))

    kpt_twin_xyz_list = []
    for icluster, vname, frameid in clip_typical_list:
        matc = osp.join(matcalib_dir, vname+'.smoothed_foot.matcalibpkl')
        assert osp.exists(matc)
        matc_data = pickle.load(open(matc, 'rb'))
        kpt_twin_xyz = matc_data['keypoints_xyz_ba'][frameid:frameid+24, ...]
        kpt_twin_xyz_list.append(kpt_twin_xyz)


    outdir = osp.join(osp.dirname(clippredpklfile), 'motifshowmulti')
    os.makedirs(outdir, exist_ok=True)
    theta = [5, 20, 45, 60, 75, 85]
    naxes = len(theta)

    for icluster, kpt_twin_xyz in zip(tqdm.tqdm(cluster_list), kpt_twin_xyz_list):
        # kpt_twin_xyz = kpt_twin_xyz[2:]
        kpt_twin_xyz = kpt_twin_xyz[[0,6, 12, 18, 23]]

        fig = plt.figure(figsize=(11, 11))
        for iax, plan_theta in enumerate(theta):
            ax = plt.subplot(naxes, 1, iax+1)
            plan_theta45 = plan_theta / 180 * np.pi
            plot_an_axes(ax, plan_theta45, kpt_twin_xyz, rot_degree=0)
            ax.set_title(f'plan_theta={ 90- plan_theta}°')

        # set all the axes to the same x-axis limits
        plt.tight_layout()
        plt.savefig(osp.join(outdir, f'{icluster}.jpg'))
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("clippredpkl", type=str)
    parser.add_argument("matcalib_dir", type=str, help='Folder of all the *smoothfoot.matcalib')
    args = parser.parse_args()
    assert osp.isfile(args.clippredpkl)
    assert osp.isdir(args.matcalib_dir)
    main(args.clippredpkl, args.matcalib_dir)
