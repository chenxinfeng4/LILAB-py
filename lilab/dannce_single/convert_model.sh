cd ~/DATA/dannce/demo_single/rat14_1280x800x9_mono/DANNCE/train_results/MAX
python -m lilab.dannce.t1_keras2onnx latest.hdf5 --batch 1
polygraphy run /home/liying_lab/chenxinfeng/ml-project/LILAB-py/lilab/tensorrt/constrained_network.py \
    --precision-constraints obey \
    --trt --fp16 --save-engine latest.engine
