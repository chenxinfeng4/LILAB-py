#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
polygraphy run constrained_network.py --precision-constraints obey \
    --trt --fp16 --load-inputs inputs_gt.json --load-outputs outputs_onnx_fp16.json \
    --atol 0.003 --save-engine fullmodel_end_fp16.engine


polygraphy run constrained_network.py --precision-constraints obey \
    --trt --tf16 \
    --atol 0.003 --save-engine fullmodel_end_fp16.engine

"""

"""
Parses an ONNX model, then adds precision constraints so specific layers run in FP32.
"""

from polygraphy import func
from polygraphy.backend.trt import NetworkFromOnnxPath
import tensorrt as trt

import logging

# 创建日志器对象
logger = logging.getLogger(__name__)

# 设置logger可输出日志级别范围
logger.setLevel(logging.DEBUG)

# 添加控制台handler，用于输出日志到控制台
console_handler = logging.StreamHandler()
# 添加日志文件handler，用于输出日志到文件中
file_handler = logging.FileHandler(filename='log.log', encoding='UTF-8')

# 将handler添加到日志器中
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Load the model, which implements the following network:
#
# x -> MatMul (I_rot90) -> Add (FP16_MAX) -> Sub (FP16_MAX) -> MatMul (I_rot90) -> out
#
# Without constraining the subgraph (Add -> Sub) to FP32, this model may
# produce incorrect results when run with FP16 optimziations enabled.
parse_network_from_onnx = NetworkFromOnnxPath("/home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x10_mono/DANNCE/train_results/MAX/fullmodel_weights/fullmodel_end.onnx")


@func.extend(parse_network_from_onnx)
def load_network(builder, network, parser):
    """The below function traverses the parsed network and constrains precisions
    for specific layers to FP32.
    See examples/cli/run/04_defining_a_tensorrt_network_or_config_manually
    for more examples using network scripts in Polygraphy.
    """
    counts = 0
    for layer in network:
        # Set computation precision for Add and Sub layer to FP32
        logger.debug(layer.name)

        if "normalization" in layer.name:
            layer.precision = trt.float32
            
        # Set the output precision for the Add layer to FP32.  Without this,
        # the intermediate output data of the Add may be stored as FP16 even
        # though the computation itself is performed in FP32.
        if "normalization" in layer.name:
            layer.set_output_type(0, trt.float32)
            counts += 1
    
    logger.debug("======found {} constrained layers".format(counts))

