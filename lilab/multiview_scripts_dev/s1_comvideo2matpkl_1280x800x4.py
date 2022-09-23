# python -m lilab.multiview_scripts_dev.s1_comvideo2matpkl_1280x800x4  A.mp4
# python -m lilab.multiview_scripts_dev.s1_ballvideo2matpkl_1280x800x4 
from lilab.multiview_scripts_dev.s1_ballvideo2matpkl_1280x800x4 import *

config = '/home/liying_lab/chenxinfeng/DATA/mmpose/res18_coco_com2d_256x256_ZJF.py'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, default=None, help='path to video or folder')
    parser.add_argument('--config', type=str, default=config)
    parser.add_argument('--checkpoint', type=str, default=checkpoint)
    arg = parser.parse_args()

    video_path, config, checkpoint = arg.video_path, arg.config, arg.checkpoint
    assert osp.exists(video_path), 'video_path not exists'
    if osp.isfile(video_path):
        video_path = [video_path]
    elif osp.isdir(video_path):
        video_path = glob.glob(osp.join(video_path, '*.mp4'))
        assert len(video_path) > 0, 'no video found'
    else:
        raise ValueError('video_path is not a file or folder')

    args_iterable = itertools.product(video_path, range(len(pos_views)))
    # init the workers pool
    # mmap_cuda.workerpool_init(range(num_gpus), MyWorker, config, checkpoint)
    # mmap_cuda.workerpool_compute_map(args_iterable)

    worker = MyWorker(config, checkpoint)
    for args in args_iterable:
        worker.compute(args)

    # post_process pkl files to matpkl
    for video in video_path:
        convert2matpkl(video)
        print('python -m lilab.multiview_scripts_new.s2_matpkl2ballpkl',
              video.replace('.mp4', '.matpkl'),
              '--time 1 2 3 4 5')
    