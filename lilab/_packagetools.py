import __main__
def get_subs():
    import os
    import os.path as osp
    parentdir = osp.dirname(osp.abspath(__main__.__file__))
    # get the files and folders in parentdir directory
    files, folders = [], []
    for f in os.listdir(parentdir):
        if osp.isfile(osp.join(parentdir, f)):
            files.append(f)
        else:
            folders.append(f)

    files = [f[:-3] for f in files if f.endswith('.py') and not f.startswith('_')] # filter out non-py files
    files = sorted(files)
    folders = [f+'/' for f in folders if not f.startswith('.') and not f.startswith('__')] # filter out useless folders
    folders = sorted(folders)
    return folders+files


def processArgs(args):
    # get the files and folders in current directory
    if args.ls:
        subfolder = get_subs()
        print('\n'.join(subfolder))
    elif args.version:
        # run the lilab._verison.py
        from ._version import __version__
        print(__version__)

def getArgs():
    import argparse
    import sys
    from ._packagetools import processArgs
    parser = argparse.ArgumentParser(description='LILAB-py')
    # args: --ls, --help -h, --version -v
    parser.add_argument('--ls', action='store_true', help='list all sub-modules')
    parser.add_argument('--version', '-v', action='store_true', help='show version')
    args = parser.parse_args()
    return args