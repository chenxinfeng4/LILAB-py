# python -m lilab.tensorrt.t3_trt_to_engine /A/B/C.trt
import torch
import tensorrt as trt
import os.path as osp
from torch2trt import TRTModule
import argparse

trt_file = '/home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x10_mono/DANNCE/train_results/MAX/fullmodel_weights/fullmodel_end_fp16.trt'


# %%
def convert(trt_file):
    assert '.trt' in trt_file
    engine_file = trt_file.replace('.trt', '.engine')
    module_dict = torch.load(trt_file)
    engine_bytes = module_dict['engine']
    with open(engine_file, 'wb') as f:
        f.write(engine_bytes)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trt', type=str)
    args = parser.parse_args()
    assert osp.exists(args.trt)
    convert(args.trt)