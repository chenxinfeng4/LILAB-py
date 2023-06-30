# python -m lilab.mmpose_dev.a2_convert_mmpose2onnx G:\mmpose\res50_coco_ball_512x512_ZJF.py --full --dynamic
# It's for top-down version of the network
import numpy as np
import torch

from mmpose.apis import init_pose_model
from mmcv import Config
import os.path as osp
import argparse
from torch.nn.parameter import Parameter
from lilab.mmpose_dev.a2_convert_mmpose2engine import findcheckpoint_pth

config = '/home/liying_lab/chenxinfeng/DATA/mmpose/res50_coco_ball_512x512_ZJF.py'
checkpoint = None

class NormalizedModel(torch.nn.Module):
    def __init__(self, mean_value, std_value, module):
        super().__init__()
        self.mean_value = Parameter(torch.Tensor(mean_value).view(1,3,1,1)*255.0, requires_grad=False)
        self.std_value = Parameter(torch.Tensor(std_value).view(1,3,1,1)*255.0, requires_grad=False)
        self.module = module
        
    def forward(self, x): #x: NCHW rgb 0-255
        y = (x - self.mean_value) / self.std_value
        y = self.module(y)
        return y

    def forward_dummy(self,x):
        return self.forward(x)


# %% read config file and get the image 
def _convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def convert(config, checkpoint, full, dynamic):
    cfg = Config.fromfile(config, checkpoint)
    output_file = osp.splitext(checkpoint)[0] + ('.full.onnx' if full else '.onnx')
    image_size = cfg.data_cfg.image_size
    
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    elif isinstance(image_size, (list, tuple)):
        image_size = tuple(image_size)
        if len(image_size) == 1:
            image_size = (image_size[0], image_size[0])
    else:
        raise TypeError('image_size must be either int or tuple of int')

    model = init_pose_model(config, checkpoint, device='cpu')
    model = _convert_batchnorm(model)
    model.forward = model.forward_dummy

    pipeline = model.cfg.data['test']['pipeline']
    pipeline_norm = [p for p in pipeline if p['type'] == 'NormalizeTensor'][0]
    mean_value = pipeline_norm['mean']
    std_value = pipeline_norm['std']

    if full:
        model = NormalizedModel(mean_value, std_value, model)

    model = model.eval()
    input_shape = (1, 3, image_size[1], image_size[0])
    if dynamic:
        dynamic_axes = {'input_1': {0: 'batch_size'}, 'output_1': {0: 'batch_size'}}
    else:
        dynamic_axes = None

    dummy_input = torch.randn(input_shape, requires_grad=True)    
    torch.onnx.export(model, dummy_input, output_file, export_params=True,
        input_names=["input_1"], output_names=["output_1"],
        dynamic_axes = dynamic_axes,
        keep_initializers_as_inputs=True, verbose=False, opset_version=11)
    print('Saving onnx model to: {}'.format(osp.basename(output_file)))

    C,H,W = input_shape[1:]
    out_enginefile = osp.splitext(output_file)[0] + '_fp16.engine'
    out_cachefile = osp.join(osp.dirname(osp.abspath(output_file)), '.cache.txt')
    if dynamic_axes:
        print(f"trtexec --onnx={output_file} "
        "--fp16 "
        f"--saveEngine={out_enginefile} "
        f"--timingCacheFile={out_cachefile} --workspace=3072 "
        f"--optShapes=input_1:4x{C}x{H}x{W} "
        f"--minShapes=input_1:1x{C}x{H}x{W} "
        f"--maxShapes=input_1:10x{C}x{H}x{W} ")
    else:
        print(f"trtexec --onnx={output_file} "
        "--fp16 "
        f"--saveEngine={out_enginefile} "
        f"--timingCacheFile={out_cachefile} --workspace=3072 ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PyTorch model to TensorRT model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--full', action='store_true', help='Use full model which integrated input normlization layer')
    parser.add_argument('--dynamic', action='store_true', help='dynamic shape')
    args = parser.parse_args()
    config = args.config
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = findcheckpoint_pth(config)
    convert(config, checkpoint, args.full, args.dynamic)
