# python -m lilab.mmlab_scripts.mmdet_videos_to_pkl videos/ config checkpoint
import argparse
import glob
import os.path as osp
import mmcv
import tqdm
from mmdet.apis import inference_detector as inference_model, init_detector as init_model
from mmdet.core import encode_mask_results
from lilab.mmlab_scripts.detpkl_to_segpkl import convert as detpkl_to_segpkl

import lilab.cvutils.map_multiprocess_cuda as mmp_cuda

class MyWorker(mmp_cuda.Worker):
    def __init__(self, config, checkpoint):
        super().__init__()
        self.model = init_model(config, checkpoint, device=f'cuda:{self.cuda}')

    def compute(self, video):
        out_pkl = osp.splitext(video)[0] + '.pkl'
        video_reader = mmcv.VideoReader(video)

        outputs = []
        for frame in tqdm.tqdm(video_reader, position=self.id, desc=f'[{self.id+1}]'):
            result = inference_model(self.model, frame)
            if len(result)==2:
                result = [result]
            result = [(bbox_results, encode_mask_results(mask_results))
                        for bbox_results, mask_results in result]
            resultf = filt_by_thr(result)
            outputs.extend(resultf)
        # save to pkl
        mmcv.dump(outputs, out_pkl) #save to det_pkl
        detpkl_to_segpkl(out_pkl)   #save to seg_pkl


def filt_by_thr(result, thr=0.5):
    result_out = []
    for a_frame_result in result:
        bbox_results, mask_results = a_frame_result
        a_frame_out = [[],[]]
        for a_class_bbox, a_class_mask in zip(bbox_results, mask_results):
            p_vals = a_class_bbox[:,-1]
            valid  = p_vals > thr
            a_class_bbox = a_class_bbox[valid]
            a_class_mask = [mask for mask,v in zip(a_class_mask,valid) if v]
            a_frame_out[0].append(a_class_bbox)
            a_frame_out[1].append(a_class_mask)

        result_out.append(a_frame_out)
    return result_out


def main(folder, config, checkpoint, num_gpus):
    videos = glob.glob(osp.join(folder, '*.mp4')) + glob.glob(
                        osp.join(folder, '*.avi'))
    mmp_cuda.workerpool_init(range(num_gpus), MyWorker, config, checkpoint)
    mmp_cuda.workerpool_compute_map(videos)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMSegementation video demo')
    parser.add_argument('videos', help='Video folder')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--num_gpus', default=4, type=int, help='Device used for inference')
    args = parser.parse_args()
    main(args.videos, args.config, args.checkpoint, args.num_gpus)

# if __name__ == '__main__':
#     main('/home/liying_lab/chenxinfeng/DATA/videobenchmark/test', 
#          '/home/liying_lab/chenxinfeng/DATA/CBNetV2/mask_rcnn_r101_fpn_2x_coco_bwrat.py',
#           '/home/liying_lab/chenxinfeng/DATA/CBNetV2/work_dirs/mask_rcnn_r101_fpn_2x_coco_bwrat/latest.pth',
#           2)
