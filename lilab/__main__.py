## main
import argparse
parser = argparse.ArgumentParser(description='LILAB-py')
# args: --ls, --help -h, --version -v
parser.add_argument('--ls', action='store_true', help='list all sub-modules')
parser.add_argument('--version', '-v', action='store_true', help='show version')
args = parser.parse_args()

if args.ls:
    subfolder = """lilab.cvutils
dlc_scripts
mmlab_scripts
multiview_scripts
    """
    print(subfolder)
elif args.version:
    # run the lilab._verison.py
    from ._version import __version__
    print(__version__)