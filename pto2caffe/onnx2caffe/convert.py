import onnx
from compare import compare
from onnx import shape_inference
from preprocess import get_input_tensor
from onnx2caffe.model import Model


def set_batch_size(onnx_model):
    inputs = onnx_model.graph.input
    for input in inputs:
        if len(input.type.tensor_type.shape.dim):
            dim = input.type.tensor_type.shape.dim[0]
            if dim.dim_value == 0:
                dim.dim_value = 1


def convert(onnx_file, caffe_model_path, param=None):
    onnx_model = onnx.load(onnx_file)
    opset = onnx_model.opset_import[0].version
    set_batch_size(onnx_model)

    # ONNX Simplifier
    if param.get('simplifier', 0) == 1:
        if opset >= 7:
            from onnxsim import simplify
            onnx_model, check = simplify(onnx_model)
            assert check, "Simplified ONNX model could not be validated"
        else:
            print('Warning: Model\'s opset Version < 7.')

    onnx_model = shape_inference.infer_shapes(onnx_model)
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)

    model = Model(onnx_model, param)
    model.parse()
    model.convert()
    model.save(caffe_model_path)

    input_tensor = get_input_tensor(param, model.inputs_shape[0], None)

    if opset >= 7:
        compare('onnx', onnx_model, caffe_model_path, input_tensor, param.get('compare', -1))
