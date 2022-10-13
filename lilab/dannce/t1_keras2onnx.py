# python t1_keras2onnx.py ./models/dannce_model.h5 | xargs -i python t2_onnx2trt.py "{}"
import argparse
from tensorflow.keras.models import load_model
from dannce.engine import nets, losses, ops
import tensorflow as tf
import onnx
import tf2onnx
import sys
import os.path as osp
modelfile = '/home/liying_lab/chenxinfeng/DATA/dannce/demo/rat14_1280x800x10_mono/DANNCE/train_results/MAX/fullmodel_weights/fullmodel_end.hdf5'


def main(modelfile):
    model = load_model(modelfile,
        custom_objects={
            "ops": ops,
            "slice_input": nets.slice_input,
            "mask_nan_keep_loss": losses.mask_nan_keep_loss,
            "mask_nan_l1_loss": losses.mask_nan_l1_loss,
            "euclidean_distance_3D": losses.euclidean_distance_3D,
            "centered_euclidean_distance_3D": losses.centered_euclidean_distance_3D,
        },
    )
    batchsize = 1
    input_signature = [tf.TensorSpec([batchsize,64,64,64,10], tf.float32, name='input_1')] if batchsize>0 else None
    onnx_model, _ = tf2onnx.convert.from_keras(model,
                    input_signature=input_signature)
    onnx_file = osp.splitext(modelfile)[0] + ('.onnx' if batchsize == 1 else f'_batch{batchsize}.onnx')
    onnx.save(onnx_model, onnx_file)
    print('Saved model to: {}'.format(onnx_file), file=sys.stderr)
    print(onnx_file)
    return onnx_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modelfile", type=str,
                        help="Path to the model file.")
    args = parser.parse_args()
    main(args.modelfile)
