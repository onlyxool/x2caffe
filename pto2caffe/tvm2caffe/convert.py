import sys
import tvm
from compare import compare2
from preprocess import get_input_tensor
from tvm2caffe.model import Model


def check_dynamic_input(onnx_model, input_shape_dict, param_input_shape):
    for index, input_shape in enumerate(inputs_shape_dict.value()):
        if None in input_shape and param_input_shape is None:
            sys.exit('Error: Dynamic Model input detected, Please Use -input_shape to overwrite input shape.' + input_str + '\n')
        elif None in input_shape:
            input_shape = param_input_shape[index]


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


def get_input_shape_dict(model, model_type):
    shape_dict = dict()
    if model_type == 'onnx':
        for index, input in enumerate(model.graph.input):
            if input.name not in [tensor.name for tensor in model.graph.initializer] + [tensor.name for tensor in model.graph.sparse_initializer]:
                shape_dict[input.name] = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
    elif model_type == 'tflite':
        for index in range(model.Subgraphs(0).InputsLength()):
            shape_dict[model.Subgraphs(0).Inputs(index)] = model.Subgraphs(0).Tensors(model.Subgraphs(0).Inputs(index)).ShapeAsNumpy().tolist()
    elif model_type == 'tensorflow':
        pass
    elif model_type == 'pytorch':
        pass

    return shape_dict


def get_input_dtype_dict(model, model_type):
    dtype_dict = dict()
    if model_type == 'tflite':
        numpy_dtype = ['float32', 'float16', 'int32', 'uint8', 'int64', 'string', 'bool', 'int16', 'complex64', 'int8', 'float64', 'complex128']
        for index in range(model.Subgraphs(0).InputsLength()):
            dtype_dict[model.Subgraphs(0).Inputs(index)] = numpy_dtype[model.Subgraphs(0).Tensors(model.Subgraphs(0).Inputs(index)).Type()]

    return dtype_dict

import pathlib
def convert(model_file, caffe_model_path, param=None):
    if pathlib.Path(model_file).suffix.lower() == '.onnx':
        import onnx

        onnx_model = onnx.load(model_file)
        inputs_shape_dict = get_input_shape_dict(onnx_model, 'onnx')
        tvm_model, tvm_model_params = tvm.relay.frontend.from_onnx(onnx_model, shape=inputs_shape_dict, freeze_params=True)
    elif pathlib.Path(model_file).suffix.lower() == '.tflite':
        import tflite

        tflite_model_buf = open(model_file, "rb").read()
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        inputs_shape_dict = get_input_shape_dict(tflite_model, 'tflite')
        inputs_dtype_dict = get_input_dtype_dict(tflite_model, 'tflite')
        tvm_model, tvm_model_params = tvm.relay.frontend.from_tflite(tflite_model, shape_dict=inputs_shape_dict, dtype_dict=inputs_dtype_dict)
    elif pathlib.Path(model_file).suffix.lower() == '.pb':
        import tensorflow as tf

        with tf.compat.v1.gfile.GFile(model_file, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            graph = tf.import_graph_def(graph_def, name="")

#       inputs_shape_dict = get_input_shape_dict(graph_def, 'tensorflow')
        tvm_model, tvm_model_params = tvm.relay.frontend.from_tensorflow(graph_def, layout=param['layout'], shape={'image_input': [1,416,416,3]})
    elif pathlib.Path(model_file).suffix.lower() == '.pt':
#       inputs_shape_dict = {:}
        tvm_model, tvm_model_params = tvm.relay.frontend.from_pytorch(model_file, input_infos, custom_convert_map=None, use_parser_friendly_name=False, keep_quantized_weight=False)

    model = Model(tvm_model, tvm_model_params, param)
    model.parse()
    model.convert()
    caffe_net = model.save(caffe_model_path)

#    inputs_tensor = list()
#    for index, input_name in enumerate(model.inputs):
#        inputs_tensor.append(get_input_tensor(param, model.inputs_shape[index], model.inputs_dtype[index], None))
#
#    if opset >= 7:
#        compare2(model, caffe_net, inputs_tensor, param.get('compare', -1))
