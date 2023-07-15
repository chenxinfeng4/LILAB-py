# %% python -m lilab.dannce.t1_keras2onnx ./models/dannce_model.h5 --batch 2
import argparse
import os
import os.path as osp
import sys

import onnx
import tensorflow as tf
import tf2onnx
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, GlobalMaxPooling3D, Reshape
import tensorflow.keras.backend as K

from dannce.engine import losses, nets, ops

modelfile = "/home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x9_mono_young/DANNCE/train_results/MAX/latest.hdf5"
batchsize = 2  # [None = Dynamic, 1|2|... = Fix shape]

from tensorflow.keras import layers

class NormalizedLayer(layers.Layer):
    def __init__(self, mean_value, std_value):
        super(NormalizedLayer, self).__init__()
        self.mean_value = mean_value
        self.std_value = std_value

    def call(self, inputs):
        return (inputs - self.mean_value) / self.std_value
    
def main(modelfile, batchsize):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use cpu to load model
    model = load_model(
        modelfile,
        custom_objects={
            "ops": ops,
            "slice_input": nets.slice_input,
            "mask_nan_keep_loss": losses.mask_nan_keep_loss,
            "mask_nan_l1_loss": losses.mask_nan_l1_loss,
            "euclidean_distance_3D": losses.euclidean_distance_3D,
            "centered_euclidean_distance_3D": losses.centered_euclidean_distance_3D,
        },
    )
    nchannel_fix = model.input.shape[-1]
    input_signature = (
        [
            tf.TensorSpec(
                [batchsize, 64, 64, 64, nchannel_fix], tf.float32, name="input_1"
            )
        ]
        if batchsize > 0
        else [
            tf.TensorSpec(
                [None, 64, 64, 64, nchannel_fix], tf.float32, name="input_1"
            )
        ]

    )
    input_layer = Input(shape=model.input_shape[1:])
    normalized_layer = NormalizedLayer(3,10)(input_layer)
    head_model = model(normalized_layer)
    output_layer_pval = GlobalMaxPooling3D()(head_model)
    reshape_layer = Reshape((-1, model.output_shape[-1]))
    argmax_layer = Lambda(lambda x: K.argmax(x, axis=-1))
    reshape_output = reshape_layer(model.output)
    output_with_argmax = argmax_layer(reshape_output)

    model = Model(inputs=input_layer, outputs=[reshape_output])
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)
    onnx_file = osp.splitext(modelfile)[0] + (
        "_full.onnx" if batchsize <= 1 else f"_batch{batchsize}_full.onnx"
    )
    onnx.save(onnx_model, onnx_file)
    print("Saved model to: {}".format(onnx_file), file=sys.stderr)
    print(onnx_file)
    return onnx_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modelfile", type=str, help="Path to the model file.")
    parser.add_argument("--batch", type=int, default=-1)
    args = parser.parse_args()
    if args.batch<=0: args.batch=-1
    main(args.modelfile, args.batch)
