# python -m lilab.dannce.p2_dataset_10view_to9view DIR
import argparse
import cv2
import glob
import os
import os.path as osp
import scipy.io as sio

vdir='/home/liying_lab/chenxinfeng/DATA/dannce/data/bw_rat_1280x800x10_2022-6-16_2_TPH2_voxel'

def convert(vdir):
    outdir=vdir.replace('1280x800x10', '1280x800x9')
    os.makedirs(outdir, exist_ok=True)
    imgfiles = glob.glob(vdir+'/*.jpg')

    for imgfile in imgfiles:
        img = cv2.imread(imgfile)
        assert img.shape==(3200, 3840, 3)
        imgcrop = img[:800*3]
        savename = osp.join(outdir, osp.basename(imgfile))
        cv2.imwrite(savename, imgcrop, [int(cv2.IMWRITE_JPEG_QUALITY),99])

    annofile = osp.join(vdir, 'anno.mat')
    matdata = sio.loadmat(annofile)
    outdata = dict(
        status = matdata['status'][:,:9,:],
        # skeleton = matdata['skeleton'],
        imageSize = matdata['imageSize'][:9],
        imageNames = matdata['imageNames'],
        data_3D = matdata['data_3D'],
        camParams = matdata['camParams'][:9]
    )
    sio.savemat(osp.join(outdir, 'anno.mat'), outdata)
    print('saved to: ', osp.join(outdir, 'anno.mat'))

    # calibpkl
    calibfile = glob.glob(vdir+'/*.calibpkl.mat')[0]
    matdata = sio.loadmat(calibfile)
    matdata['ba_poses'] = matdata['ba_poses'][:,:9]
    sio.savemat(osp.join(outdir, osp.basename(calibfile)), matdata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vdir', type=str)
    args = parser.parse_args()
    vdir = args.vdir
    assert osp.exists(vdir), 'pklfile not exists'
    convert(vdir)
