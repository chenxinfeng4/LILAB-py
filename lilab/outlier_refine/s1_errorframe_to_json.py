# python -m lilab.outlier_refine.s1_errorframe_to_json /A/B/C/
# %%
import tqdm
import json
import os.path as osp
import glob
import collections
import argparse
import re
# %%
def get_num(img):
    dirname = osp.dirname(osp.abspath(img))
    basename = osp.basename(osp.abspath(img))
    nakename, ext = osp.splitext(basename)
    ind = nakename.rfind('_')
    assert '.mp4' in nakename
    vname = re.findall(r'.*mp4',nakename)[0]
    frameid = re.findall(r'.*mp4(.*)$',nakename)[0]
    frameid = re.findall(r'(\d+)$', frameid)[0]
    assert frameid.isdigit()
    frameid = int(frameid)
    videoname = vname
    return videoname, frameid


def get_numB(img):
    dirname = osp.dirname(osp.abspath(img))
    basename = osp.basename(osp.abspath(img))
    nakename, ext = osp.splitext(basename)
    ind = nakename.rfind('_')
    vname = nakename[:ind]
    frameid = nakename[ind+1:]
    assert frameid.isdigit()
    frameid = int(frameid)
    videoname = vname + '.mp4'
    return videoname, frameid


# %%
dir = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/wtxwt_social/picked/'

def extract_json(dir):
    findext = lambda x: glob.glob(osp.join(dir, x))
    error_imgs = findext('*.png') + findext('*.jpg')

    outdict = collections.defaultdict(list)
    for error_img in tqdm.tqdm(error_imgs):
        videoname, frameid = get_numB(error_img)
        outdict[videoname].append(frameid)

    outjson = osp.join(dir, 'out.json')
    with open(outjson, 'w') as f:
        json.dump(outdict, f, indent=4)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('dir', type=str)
    args = argparser.parse_args()
    extract_json(args.dir)
