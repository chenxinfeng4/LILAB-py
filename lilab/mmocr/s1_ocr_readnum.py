# conda activate mmdet
"""
from lilab.mmocr.s1_ocr_readnum import MyOCR
myOCR = MyOCR()
pred = myOCR(imgfile)
"""
#
import os.path as osp
import mmocr
from mmocr.apis import init_detector, model_inference
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401

config_ = '/home/liying_lab/chenxinfeng/DATA/mmocr/sar_num.py'
def find_checkpoint(config):
    basedir = osp.dirname(config)
    basenakename = osp.splitext(osp.basename(config))[0]
    checkpoint = osp.join(basedir, 'work_dirs', basenakename, 'latest.pth')
    assert osp.isfile(checkpoint), 'checkpoint not found: {}'.format(checkpoint)
    return checkpoint


class MyOCR:
    def __init__(self, config=None, checkpoint=None):
        if config is None:
            config = config_
        if checkpoint is None:
            checkpoint = find_checkpoint(config)
        device='cuda'

        model = init_detector(config, checkpoint, device=device)
        self.model = model
        if hasattr(model, 'module'):
            model = model.module
    
    def __call__(self, img):
        result = model_inference(self.model, img)
        pred_label = result['text']
        return pred_label
