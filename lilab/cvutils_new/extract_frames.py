# python -m lilab.cvutils_new.extract_frames testA.mp4
import argparse
import os
import cv2
import numpy as np
import tqdm
import os.path as osp
import sys
import glob
import ffmpegcv

numframe_to_extract = 100
maxlength = 27000 #30000
frame_dir = "outframes"
frame_min_interval = 30

def extract(video_input, numframe_to_extract, maxlength):
    dirname,filename=os.path.split(video_input)
    nakefilename = os.path.splitext(filename)[0]
    cap = ffmpegcv.VideoCaptureNV(video_input, gpu=2)
    os.makedirs(os.path.join(dirname, frame_dir), exist_ok = True)
    length = cap.count
    length = min([maxlength, length-1])
    downsample_length = length // frame_min_interval
    np.random.seed(0)
    idxframe_to_extract = set(np.random.permutation(downsample_length)[:numframe_to_extract]*frame_min_interval + 50)
    idxframe_max = max(idxframe_to_extract)

    # for i in tqdm.tqdm(range(10000)):
    #     ret, frame = cap.read()

    for iframe in tqdm.tqdm(range(length)):
        ret, frame = cap.read()
        if not ret: break
        if iframe>idxframe_max: break
        if iframe not in idxframe_to_extract: continue
        filename = os.path.join(dirname, frame_dir, nakefilename + '_{0:06}.jpg'.format(iframe))
        cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY),100])
        
    cap.release()
    

def extract_iview(video_input, iview, numframe_to_extract, maxlength):
    from lilab.cameras_setup import get_view_xywh_1280x800x10 as get_view_xywh
    import ffmpegcv
    crop_xywh = get_view_xywh()[iview]
    dirname,filename=os.path.split(video_input)
    nakefilename = os.path.splitext(filename)[0]
    cap = ffmpegcv.VideoCaptureNV(video_input, crop_xywh=crop_xywh)
    os.makedirs(os.path.join(dirname, frame_dir), exist_ok = True)
    length = len(cap)
    length = min([maxlength, length-1])
    downsample_length = length // frame_min_interval
    idxframe_to_extract = set(np.random.permutation(downsample_length)[:numframe_to_extract]*frame_min_interval)
    idxframe_max = max(idxframe_to_extract)

    for iframe in tqdm.tqdm(range(length)):
        ret, frame = cap.read()
        if not ret: break
        if iframe>idxframe_max: break
        if iframe not in idxframe_to_extract: continue
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        filename = os.path.join(dirname, frame_dir, nakefilename + '_{0:06}_output_{1}.jpg'.format(iframe, iview))
        cv2.imwrite(filename, frame)
        
    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract image')
    parser.add_argument('video_path', type=str, default=None, help='path to image or folder')
    parser.add_argument('--iview', type=int, default=None)
    parser.add_argument('--npick', type=int, default=numframe_to_extract)
    args = parser.parse_args()

    video_path = args.video_path
    assert osp.exists(video_path), 'video_path not exists'
    if osp.isfile(video_path):
        video_path = [video_path]
    elif osp.isdir(video_path):
        video_path = [f for f in glob.glob(osp.join(video_path, '*.mp4'))
                        if f[-4] not in '0123456789']
        assert len(video_path) > 0, 'no video found'
    else:
        raise ValueError('video_path is not a file or folder')

    # read config_video.py
    for video_input in video_path:
        if args.iview is None:
            extract(video_input, args.npick, maxlength)
        else:
            extract_iview(video_input, args.iview, args.npick, maxlength)
    
    print("Succeed")
