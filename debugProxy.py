import sys
import runpy
import os

os.chdir('/mnt/liying.cibr.ac.cn_Data_Temp/multiview_9/chenxf/LTZxWT_230505/ball/')
# args = 'python -m lilab.multiview_scripts_new.s2_matpkl2ballpkl /mnt/liying.cibr.ac.cn_Data_Temp/multiview-large/TPH2KOxWT/2022-06-16ball.matpkl --time 1 9 17 23 27'
# args = 'python -m lilab.metric_seg.s3_cocopkl_vs_cocopkl --gt_pkls /home/liying_lab/chenxinfeng/DATA/CBNetV2/data/rats_metric/te1/intense_pannel.cocopkl --pred_pkls /home/liying_lab/chenxinfeng/DATA/CBNetV2/data/rats_metric/te2/intense_pannel.cocopkl '
args = 'python -m lilab.multiview_scripts_dev.p2_calibpkl_refine_byglobal 2023-05-05_22-51-09Sball.calibpkl 2023-05-05_22-51-09Sball.globalrefpkl'

# args = 'python test.py 5 7'

args = args.split()
if args[0] == 'python':
    """pop up the first in the args""" 
    args.pop(0)

if args[0] == '-m':
    """pop up the first in the args"""
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path

sys.argv.extend(args[1:])

fun(args[0], run_name='__main__')
