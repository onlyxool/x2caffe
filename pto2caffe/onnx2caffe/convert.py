import onnx
from onnx import shape_inference
from onnx2caffe.model import Model
from caffe_dump import dump_caffe_model
from compare import compare


def set_batch_size(onnx_model):
    inputs = onnx_model.graph.input
    for input in inputs:
        dim = input.type.tensor_type.shape.dim[0]
        if dim.dim_value == 0:
            dim.dim_value = 1


def convert(onnx_file, input_tensor, caffe_model_path, dump_level=-1, param=None):
    onnx_model = onnx.load(onnx_file)

    set_batch_size(onnx_model)

    # ONNX Simplifier
    if param.get('simplifier', 0) == 1:
        from onnxsim import simplify
        onnx_model, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"

    onnx_model = shape_inference.infer_shapes(onnx_model)
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)

    model = Model(onnx_model, param)
    model.parse()
    model.convert()
    model.save(caffe_model_path)

    if dump_level >= 0:
        model.dump(onnx_model, param['model_name'], input_tensor, dump_level)

    if dump_level == 3:
        dump_caffe_model(caffe_model_path, input_tensor, param['input_file'])

    compare('onnx', onnx_model, caffe_model_path, input_tensor, param.get('compare', -1))
