import sys
import onnx
from compare import compare
from preprocess import get_input_tensor
from onnx2caffe.model import Model


def check_dynamic_input(onnx_model, input_shape):
    inputs_id = list()
    for index, input in enumerate(onnx_model.graph.input):
        if input.name not in [tensor.name for tensor in onnx_model.graph.initializer] + [tensor.name for tensor in onnx_model.graph.sparse_initializer]:
            inputs_id.append(index)

    if input_shape is not None:
        if len(inputs_id) > 1:
            for index, input_id in enumerate(inputs_id):
                for i, dim in enumerate(onnx_model.graph.input[input_id].type.tensor_type.shape.dim):
                    dim.dim_value = input_shape[index][i]
        else:
             for i, dim in enumerate(onnx_model.graph.input[0].type.tensor_type.shape.dim):
                dim.dim_value = input_shape[i]
    else:
        input_str = str()
        input_dict = dict()
        for input in onnx_model.graph.input:
            input_shape = list()
            for dim in input.type.tensor_type.shape.dim:
                input_shape.append(dim.dim_value)
            if input_shape.count(0) > 0:
                input_str = input_str + ' ' + input.name + str(input_shape)

        if len(input_str) > 0:
            sys.exit('Error: Dynamic Model input detected, Please Use -input_shape to overwrite input shape.' + input_str + '\n')


def convert(onnx_file, caffe_model_path, param=None):
    onnx_model = onnx.load(onnx_file)
    opset = onnx_model.opset_import[0].version

    check_dynamic_input(onnx_model, param['input_shape'])


    # ONNX Simplifier
    if param.get('simplifier', 0) == 1:
        if opset >= 7:
            from onnxsim import simplify
            onnx_model, check = simplify(onnx_model)
            assert check, "Simplified ONNX model could not be validated"
        else:
            print('Warning: Model\'s opset Version < 7.')
    else:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)

    model = Model(onnx_model, param)
    model.parse()
    model.convert()
    caffe_net = model.save(caffe_model_path)

    inputs_tensor = list()
    for index, input_name in enumerate(model.inputs):
        inputs_tensor.append(get_input_tensor(param, model.inputs_shape[index], model.inputs_dtype[index], None))

    if opset >= 7:
        compare(model, caffe_net, inputs_tensor, param.get('compare', -1))
