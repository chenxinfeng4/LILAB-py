# concat two video file by ffmpeg
import os
import os.path as osp
import argparse
import cv2

def concat(*videopaths):
    h_v_dict = {2:[2,1], 3:[2,2], 4:[2,2], 5:[3,2], 6:[3,2], 7:[3,3], 8:[3,3], 9:[3,3]}
    n_videos = len(videopaths)
    h, v = h_v_dict[n_videos]
    vhlist = []
    for iv in range(v):
        for ih in range(h):
            pos = '+'.join(['0']+['9+w0']*ih) + '_' + '+'.join(['0']+['9+h0']*iv)
            vhlist.append(pos)
    vhlist = vhlist[:n_videos]
    whmatrix = '|'.join(vhlist).replace('0+','')
    filterstr = f"xstack=inputs={n_videos}:{whmatrix}:fill=green"

    # input lists
    input_list = ' '.join([f'-i "{videopath}"' for videopath in videopaths])

    # output file
    output_path = osp.join(osp.dirname(videopaths[0]), "concat.mp4")        
            
    # concat the videos horizontally
    os.system(f'ffmpeg {input_list} -filter_complex "{filterstr}" -y -c:v libx264  "{output_path}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # arg parse the multiple videos path
    parser.add_argument("videopaths", type=str, nargs='+')
    args = parser.parse_args()
    assert len(args.videopaths) > 1, "need at least two videos to concat"
    assert len(args.videopaths) < 10, "need at most 9 videos to concat"
    concat(*args.videopaths)