# python -m lilab.tensorrt.t3_engine_to_trt xxx.engine
# %%
import torch
import tensorrt as trt
import os.path as osp
from torch2trt import TRTModule
import argparse

engine_file = '/home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x10_mono/DANNCE/train_results/MAX/fullmodel_weights/fullmodel_end_fp16.engine'


# %%
def convert(engine_file):
    assert '.engine' in engine_file
    trtfile = engine_file.replace('.engine', '.trt')
    engine_bytes = open(engine_file, 'rb').read()
    trt_runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = trt_runtime.deserialize_cuda_engine(engine_bytes)
    input_names = [n for n in engine if engine.binding_is_input(n)]
    output_names = [n for n in engine if not engine.binding_is_input(n)]
    module_trt = TRTModule(engine, input_names, output_names)
    torch.save(module_trt.state_dict(), trtfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('engine', type=str)
    args = parser.parse_args()
    assert osp.exists(args.engine)
    convert(args.engine)