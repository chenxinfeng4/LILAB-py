python /home/liying_lab/chenxinfeng/DATA/dannce/t1_keras2onnx.py latest.hdf5

polygraphy run constrained_network.py --precision-constraints obey \
    --trt --fp16 --save-engine latest_fp16.engine \
    --load-tactics .tactics_fp16
    # --save-tactics .tactics_fp16  #631sec

# dynamic shape
polygraphy run constrained_network.py --precision-constraints obey \
    --trt --fp16 --save-engine latest_fp16.engine \
    --trt-min-shapes 1x64x64x64x4 \
    --trt-opt-shapes 1x64x64x64x4 \
    --trt-max-shapes 2x64x64x64x4

trtexec --onnx=latest.onnx --saveEngine=latest_fp32.engine \
    --timingCacheFile=.timingcache_fp32

ln -s latest_fp16.engine latest.engine
ln -s latest_fp32.engine latest.engine
