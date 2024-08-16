# python -m lilab.mmdet_dev.filter_vname
# ls *.mp4 | python -m lilab.mmdet_dev.filter_vname -p
import os.path as osp
import re
import os

pattern = ".*400p.*|.*mask.*|.*com3d.*|.*skt.*"


def filter_vname(vlist):
    repattern = re.compile(pattern)
    filtered_files = [
        file for file in vlist if file and not re.match(repattern, osp.basename(file))
    ]
    return filtered_files


if __name__ == "__main__":
    import glob
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--p", action="store_true")
    args = parser.parse_args()
    if not args.p:
        pwd = os.getcwd()
        vlist = glob.glob("*.mp4")
    else:
        vlist = sys.stdin.readlines()
        vlist = [file.rstrip() for file in vlist]

    filtered_files = filter_vname(vlist)
    for file in filtered_files:
        print(file)
