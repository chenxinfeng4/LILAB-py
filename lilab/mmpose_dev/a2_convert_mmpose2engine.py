# python -m lilab.mmpose_dev.a2_convert_mmpose2engine XXX.config

# It's for top-down version of the network
import numpy as np
import torch

from torch2trt import torch2trt
from mmcv import Config
import os.path as osp
import argparse
from torch.nn.parameter import Parameter

config = '/home/liying_lab/chenxinfeng/DATA/mmpose/res50_coco_ball_512x320_ZJF.py'
checkpoint = None

class NormalizedModel(torch.nn.Module):
    def __init__(self, mean_value, std_value, module):
        super().__init__()
        self.mean_value = Parameter(torch.Tensor(mean_value).view(1,3,1,1)*255.0, requires_grad=False)
        self.std_value = Parameter(torch.Tensor(std_value).view(1,3,1,1)*255.0, requires_grad=False)
        self.module = module
        
    def forward(self, x): #x: NCHW rgb 0-255
        # if x.shape[1] == 1:
        #     x = x.repeat(1,3,1,1)
        y = (x - self.mean_value) / self.std_value
        y = self.module(y)
        return y

    def forward_dummy(self,x):
        return self.forward(x)


def findcheckpoint_trt(config, trtnake='latest.engine'):
    """Find the latest checkpoint of the model."""
    basedir = osp.dirname(config)
    basenakename = osp.splitext(osp.basename(config))[0]
    checkpoint = osp.join(basedir, 'work_dirs', basenakename, trtnake)
    assert osp.isfile(checkpoint), 'checkpoint not found: {}'.format(checkpoint)
    return checkpoint


def findcheckpoint_pth(config):
    """Find the latest checkpoint of the model."""
    basedir = osp.dirname(config)
    basenakename = osp.splitext(osp.basename(config))[0]
    checkpoint = osp.join(basedir, 'work_dirs', basenakename, 'latest.pth')
    assert osp.isfile(checkpoint), 'checkpoint not found: {}'.format(checkpoint)
    return checkpoint


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


def convert(config, checkpoint):
    from mmpose.apis import init_pose_model
    cfg = Config.fromfile(config, checkpoint)
    output = osp.splitext(checkpoint)[0] + '.full.engine'
    image_size = cfg.data_cfg.image_size
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    elif isinstance(image_size, (list, tuple)):
        image_size = tuple(image_size)
        if len(image_size) == 1:
            image_size = (image_size[0], image_size[0])
    else:
        raise TypeError('image_size must be either int or tuple of int')

    input_shape = (1, 3, image_size[1], image_size[0])
    opt_shapes = (6, 3, image_size[1], image_size[0])
    max_shapes = (9, 3, image_size[1], image_size[0])
    model = init_pose_model(config, checkpoint, device='cpu')
    model = _convert_batchnorm(model)
    model.forward = model.forward_dummy

    pipeline = model.cfg.data['test']['pipeline']
    pipeline_norm = [p for p in pipeline if p['type'] == 'NormalizeTensor'][0]
    mean_value = pipeline_norm['mean']
    std_value = pipeline_norm['std']
    model = NormalizedModel(mean_value, std_value, model)

    with torch.cuda.device(0):
        model = model.eval().cuda()
        one_img = torch.randn(input_shape).cuda()
        out = model(one_img)
        trt_model = torch2trt(model, [one_img], 
                              min_shapes=[input_shape],
                              opt_shapes=[opt_shapes],
                              max_shapes=[max_shapes],
                              fp16_mode=False)
        print('min shapes:', trt_model(torch.randn(input_shape).cuda()).shape)
        print('opt shapes:', trt_model(torch.randn(opt_shapes).cuda()).shape)
        print('max shapes:', trt_model(torch.randn(max_shapes).cuda()).shape)
        

    print('Saving TRT model to: {}'.format(osp.basename(output)))
    trt_model.save_to_engine(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PyTorch model to TensorRT model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    args = parser.parse_args()
    config = args.config
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = findcheckpoint_pth(config)
    convert(config, checkpoint)
