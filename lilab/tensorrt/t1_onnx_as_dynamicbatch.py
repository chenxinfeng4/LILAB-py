# python -m lilab.tensorrt.t1_onnx_as_dynamicbatch xxx.onnx
import argparse
import onnx
import os.path


def change_input_dim(model):
    # Use some symbolic name not used for any other dimension
    sym_batch_dim = "N"
    # or an actal value
    actual_batch_dim = 4 

    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim 
    inputs = model.graph.input
    for input in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        # update dim to be a symbolic value
        dim1.dim_param = sym_batch_dim
        # or update it to be an actual value:
        # dim1.dim_value = actual_batch_dim


def main(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    onnx.checker.check_model(model)
    onnx.save(model, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help='input onnx model')
    args = parser.parse_args()

    outfile = os.path.splitext(args.infile)[0] + '_dynamic.onnx'
    main(change_input_dim, args.infile, outfile)
