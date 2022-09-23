# %%
import torch
import tqdm
import tensorrt as trt
from torch2trt import TRTModule
from torch2trt.torch2trt import torch_dtype_from_trt

engine_file = 'fullmodel_end_fp16.engine'

engine_bytes = open(engine_file, 'rb').read()
trt_runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
engine = trt_runtime.deserialize_cuda_engine(engine_bytes)
input_names = [n for n in engine if engine.binding_is_input(n)]
output_names = [n for n in engine if not engine.binding_is_input(n)]
trt_model = TRTModule(engine, input_names, output_names)

dtype = torch_dtype_from_trt(trt_model.engine.get_binding_dtype(0))  #float32
shape = tuple(trt_model.context.get_binding_shape(0))     #1batchx64x64x64x6camera


for _ in tqdm.tqdm():
    X_torch = torch.random(shape, dtype=dtype).cuda()
    y = trt_model(X_torch)

