cd ~/DATA/dannce/demo_single/rat14_1280x800x9_mono/DANNCE/train_results/MAX
cd ~/DATA/dannce/demo/rat14_1280x800x9_mono_young/DANNCE/train_results/MAX
python -m lilab.dannce.t1_keras2onnx latest.hdf5
polygraphy run /home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/tensorrt/constrained_network.py \
    --precision-constraints obey \
    --input-shapes input_1:[2,64,64,64,9]\
    --trt --fp16 --save-engine latest.engine

# trtexec --loadEngine=latest.engine --shapes=input_1:2x64x64x64x9
exit
polygraphy run /home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/tensorrt/constrained_network.py \
    --precision-constraints obey \
    --input-shapes input_1:[2,64,64,64,9]\
    --trt-min-shapes input_1:[1,64,64,64,9] \
    --trt-max-shapes input_1:[4,64,64,64,9] \
    --trt-opt-shapes input_1:[2,64,64,64,9] \
    --trt --fp16 --save-engine latest_dynamic.engine
