# from lilab.cvutils_new.parse_name_of_extract_frames import parsenames_indir
# python -m lilab.cvutils_new.parse_name_of_extract_frames /A/B/C/
# %%
import os.path as osp
import glob
import json
import argparse
dir = '/mnt/liying.cibr.ac.cn_Data_Temp/multiview_color/20220613-side6-addition/2022-2-24-side6-bwrat-shank3/outframes_white/'

# %%
def parsename(imagebasename):
    imagenakename = osp.splitext(imagebasename)[0]
    if '_ratwhite' in imagenakename :
        rat_id = 1
        imagenakename = imagenakename.replace('_ratwhite', '')
    elif '_ratblack' in imagenakename:
        rat_id = 0
        imagenakename = imagenakename.replace('_ratblack', '')
    else:
        rat_id = None

    assert 'output' not in imagenakename, 'output is not allowed in the name'

    framestr = imagenakename.split('_')[-1]
    frameid = int(framestr)
    videoname = imagenakename.replace(f'_{framestr}', '')+'.mp4'
    return videoname, rat_id, frameid


# %%
def parsenames_indir(dir):
    imagefiles = glob.glob(osp.join(dir,'*.jpg')) + glob.glob(osp.join(dir,'*.png'))
    imagebasenames = [osp.basename(x) for x in imagefiles]
    parsednames = [parsename(imagebasename) for imagebasename in imagebasenames]
    return imagebasenames, parsednames


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('dir', type=str)
    args = argparser.parse_args()
    imagebasenames, parsednames = parsenames_indir(args.dir)
    outdict = {'imagebasenames': imagebasenames, 'parsednames': parsednames}
    outjson = osp.join(dir, 'out_id.json')
    with open(outjson, 'w') as f:
        json.dump(outdict, f, indent=4)
