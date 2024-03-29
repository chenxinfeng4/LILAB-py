# Transform onnx to engine as fp16, freeze some layers to fp32.
polygraphy run constrained_network.py --precision-constraints obey \
    --trt --fp16 --save-engine latest_fp16.engine


# For savety, we can use fp32
polygraphy run latest.onnx --trt --onnxrt \
    --tf32 --save-engine latest.engine --atol 0.001 --rtol 0.001


# Transform onnx to engine as fp16 without freezing
MODEL="latest.full"
trtexec --onnx=${MODEL}.onnx --saveEngine=${MODEL}_fp16.engine \
    --timingCacheFile=.cache.txt \
    --explicitBatch --fp16 --workspace=2048

trtexec --onnx=fullmodel_end.onnx --saveEngine=fullmodel_end_fp32.engine \
    --timingCacheFile=.cache.txt \
    --explicitBatch --workspace=2048