# %%
import torch
import tqdm
import tensorrt as trt
from torch2trt import TRTModule
from torch2trt.torch2trt import torch_dtype_from_trt
import time

engine_file = '/home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x10_mono/DANNCE/train_results/MAX/latest.engine'

trt_model = TRTModule()
trt_model.load_from_engine(engine_file)

# %%
dtype = torch_dtype_from_trt(trt_model.engine.get_binding_dtype(0))  #float32
shape = tuple(trt_model.context.get_binding_shape(0))     #1batchx64x64x64x6camera

X_torch = torch.rand(shape, dtype=dtype).cuda()
for _ in tqdm.trange(1000):
    y = trt_model(X_torch)
    time.sleep(1/90)

y[0,0,0,0]
# %%
y = trt_model(X_torch)


for _ in tqdm.trange(1000):
    y = trt_model(X_torch)

for _ in tqdm.trange(1000):
    time.sleep(1/90)

for _ in tqdm.trange(1000):
    y = trt_model(X_torch)
    
    # time.sleep(1/50)
    torch.cuda.current_stream().synchronize()

import numpy as np
x1 = np.random.rand(1,64,64,64,10).astype('float32')
x2 = np.random.rand(1,64,64,64,10).astype('float32')

for _ in tqdm.trange(1000):
    y = np.concatenate([x1,x2])
    y_cuda = torch.from_numpy(y)

for _ in tqdm.trange(1000):
    x3 = np.empty((2,64,64,64,10), dtype=x1.dtype)
    x3[0] = x1[0]
    x3[1] = x2[0]
    y_cuda = torch.from_numpy(x3).cuda().float()

for _ in tqdm.trange(1000):
    x3 = np.empty((2,64,64,64,10), dtype=x1.dtype)
    y_cuda= torch.cat([torch.from_numpy(x1).cuda(),
                        torch.from_numpy(x2).cuda()])
    y_cuda = torch.from_numpy(x3).cuda().float()