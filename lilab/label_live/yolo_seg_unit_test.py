# conda activate mmdet; python /home/liying_lab/chenxf/ml-project/LILAB-py/lilab/yolo_seg/fun_main.py
import numpy as np
import multiprocessing
from lilab.label_live.t1_realtime_position import main as seg_main
from lilab.yolo_seg.common_variable import NFRAME, create_shared_arrays


if __name__ == "__main__":
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue(maxsize=(NFRAME - 4))
    (
        shared_array_imgNNHW,
        shared_array_com2d,
        shared_array_previ,
        shared_array_timecode,
    ) = create_shared_arrays()
    process = ctx.Process(
        target=seg_main,
        args=(
            shared_array_imgNNHW,
            shared_array_com2d,
            shared_array_previ,
            shared_array_timecode,
            q,
        ),
    )
    process.start()

    while True:
        q.get()
