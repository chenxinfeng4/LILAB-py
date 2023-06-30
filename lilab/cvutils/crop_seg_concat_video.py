from . import crop_seg_video
from . import concat_videopro as concat_video
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("videopaths", type=str, nargs='+')
    args = parser.parse_args()
    assert len(args.videopaths) > 1, "need at least two videos to concat"
    assert len(args.videopaths) < 10, "need at most 9 videos to concat"
    xylen = 400
    videopaths_seg = [crop_seg_video.main(videopath, xylen) 
                            for videopath in args.videopaths]

    videocat = concat_video.concat(*videopaths_seg)
