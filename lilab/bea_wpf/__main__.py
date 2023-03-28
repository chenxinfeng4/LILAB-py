# python -m lilab.bea_wpf matcalibpkl_or_dir
import argparse
import runpy
import sys
import os.path as osp

parser = argparse.ArgumentParser('Convert the matcalibpkl to BeA_WPF.')
parser.add_argument('matcalibpkl_or_dir', type=str)
args = parser.parse_args()

matcalibpkl_or_dir = args.matcalibpkl_or_dir
sys.argv = ['', matcalibpkl_or_dir]
runpy.run_module(mod_name='lilab.bea_wpf.s1_matcalibpkl_to_bea_3d', run_name='__main__')

if osp.isdir(matcalibpkl_or_dir):
    bea_projectdir = osp.join(matcalibpkl_or_dir, 'BeA_WPF')
elif osp.isfile(matcalibpkl_or_dir):
    bea_projectdir = osp.join(osp.dirname(matcalibpkl_or_dir), 'BeA_WPF')
else:
    raise ValueError('%s is neither a directory nor a matcalibpkl file.' % matcalibpkl_or_dir)

sys.argv = ['', bea_projectdir]
runpy.run_module(mod_name='lilab.bea_wpf.s2_create_bea_project', run_name='__main__')
