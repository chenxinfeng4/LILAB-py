# Transform onnx to engine as fp16, freeze some layers to fp32.
polygraphy run constrained_network.py --precision-constraints obey \
    --trt --fp16 --save-engine fullmodel_end_fp16.engine


# For savety, we can use fp32
polygraphy run fullmodel_end.onnx --trt --onnxrt \
    --tf32 --save-engine fullmodel_end_fp32.engine --atol 0.001 --rtol 0.001


# Transform onnx to engine as fp16 without freezing
trtexec --onnx=fullmodel_end.onnx --saveEngine=fullmodel_end_fp16.engine \
    --timingCacheFile=.cache.txt \
    --explicitBatch --fp16 --workspace=2048

trtexec --onnx=fullmodel_end.onnx --saveEngine=fullmodel_end_fp32.engine \
    --timingCacheFile=.cache.txt \
    --explicitBatch --workspace=2048