from . import crop_seg_video
from . import concat_video
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='auto crop seged video')
    parser.add_argument('videopath1', type=str, default='', help='path to video')
    parser.add_argument('videopath2', type=str, default='', help='path to video')
    args = parser.parse_args()
    xylen = 400
    videocrop1 = crop_seg_video.main(args.videopath1, xylen)
    videocrop2 = crop_seg_video.main(args.videopath2, xylen)
    videocat = concat_video.concat(videocrop1, videocrop2)
