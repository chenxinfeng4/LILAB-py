# trtexec --onnx=temp_onnx_model.onnx --saveEngine=temp_onnx_model.trt --fp16
# python t2_onnx2trt.py temp_onnx_model.onnx --fp16
# %%
import time
import argparse
import torch
import tensorrt as trt
import os.path as osp
import shutil
from torch2trt import TRTModule


onnxfile = 'temp_onnx_model.onnx'


def onnx2trt(onnx_file,
              log_level=trt.Logger.ERROR,
              max_batch_size=1,
              fp16_mode=False,
              max_workspace_size=1<<25,
              strict_type_constraints=False,
              keep_network=True,
              default_device_type=trt.DeviceType.GPU,
              dla_core=0,
              gpu_fallback=True,
              device_types={},
              **kwargs):

    # capture arguments to provide to context
    kwargs.update(locals())
    kwargs.pop('kwargs')

    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    # run once to get num outputs
    with open(onnx_file, 'rb') as f:
        onnx_bytes = f.read()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    parser.parse(onnx_bytes)


    # set max workspace size
    if max_workspace_size:
        config.max_workspace_size = max_workspace_size

    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)

    builder.max_batch_size = max_batch_size

    config.default_device_type = default_device_type
    if gpu_fallback:
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    config.DLA_core = dla_core
    
    if strict_type_constraints:
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)


    engine = builder.build_engine(network, config)

    input_names = [n for n in engine if engine.binding_is_input(n)]
    output_names = [n for n in engine if not engine.binding_is_input(n)]
    module_trt = TRTModule(engine, input_names, output_names)

    if keep_network:
        module_trt.network = network

    return module_trt


def main(onnxfile, fp16_mode=False):
    trtfile  = osp.splitext(onnxfile)[0] + '.trt'
    trtfilefull = trtfile + ('_fp16' if fp16_mode else '_fp32')
    tic = time.time()
    model_trt = onnx2trt(onnxfile, log_level=trt.Logger.WARNING, fp16_mode=fp16_mode)
    torch.save(model_trt.state_dict(), trtfile)
    toc = time.time() - tic

    shutil.copyfile(trtfile, trtfilefull)
    print('The onnx2trt takes {}:{}:{}'.format(int(toc/3600), int((toc%3600)/60), int((toc%3600)%60)))
    print('The onnx2trt saved to {}'.format(trtfile))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx', type=str)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()
    main(args.onnx, args.fp16)

#%%
if False:
    import torch
    from torch2trt import TRTModule
    with torch.cuda.device(0):
        trt_model = TRTModule()
        trt_model.load_state_dict(torch.load('temp_onnx_modelf16.trt'))
        input = torch.randn(1, 64, 64, 64, 6).cuda()
        output = trt_model(input)
        print(output.shape)