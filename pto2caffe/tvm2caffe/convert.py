import os
import sys
import tvm
import pathlib

from compare import compare
from preprocess import get_input_tensor
from tvm2caffe.model import Model
from util import shape_map_nhwc2nchw


def get_input_shape_dict(model, model_type, param):
    shape_dict = dict()
    if model_type == 'onnx':
        for index, input in enumerate(model.graph.input):
            if input.name not in [tensor.name for tensor in model.graph.initializer] + [tensor.name for tensor in model.graph.sparse_initializer]:
                shape_dict[input.name] = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
    elif model_type == 'tflite':
        for index in range(model.Subgraphs(0).InputsLength()):
            shape_dict[model.Subgraphs(0).Tensors(model.Subgraphs(0).Inputs(index)).Name().decode()] = model.Subgraphs(0).Tensors(model.Subgraphs(0).Inputs(index)).ShapeAsNumpy().tolist()
    elif model_type == 'tensorflow':
        import tensorflow as tf
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(model, name='')
        for op in graph.get_operations():
            if op.type == 'Placeholder' and 'unused_control_flow_input' not in op.outputs[0].name:
                if op.outputs[0].shape.is_fully_defined():
                    shape_dict[op.outputs[0].name.replace(':0', '')] = op.outputs[0].shape.as_list()
                else:
                    shape_dict[op.outputs[0].name.replace(':0', '')] = None
    elif model_type == 'pytorch':
        pass

    if param['input_shape'] is not None:
        for index, (input_name, input_shape) in enumerate(shape_dict.items()):
            if len(shape_dict.keys()) > 1:
                shape_dict[input_name] = param['input_shape'][index]
            else:
                shape_dict[input_name] = param['input_shape']

    for (key, value) in shape_dict.items():
        if None in value:
            print(key, value)
            sys.exit('Error: Dynamic Model input detected, Please Use -input_shape to overwrite input shape.\n')

    return shape_dict


def get_input_dtype_dict(model, model_type):
    dtype_dict = dict()
    if model_type == 'tflite':
        numpy_dtype = ['float32', 'float16', 'int32', 'uint8', 'int64', 'string', 'bool', 'int16', 'complex64', 'int8', 'float64', 'complex128']
        for index in range(model.Subgraphs(0).InputsLength()):
            dtype_dict[model.Subgraphs(0).Tensors(model.Subgraphs(0).Inputs(index)).Name().decode()] = numpy_dtype[model.Subgraphs(0).Tensors(model.Subgraphs(0).Inputs(index)).Type()]

    return dtype_dict


def convert(model_file, caffe_model_path, param=None):
    if pathlib.Path(model_file).suffix.lower() == '.onnx':
        import onnx
        onnx_model = onnx.load(model_file)
        inputs_shape_dict = get_input_shape_dict(onnx_model, 'onnx', param)
        tvm_model, tvm_model_params = tvm.relay.frontend.from_onnx(onnx_model, shape=inputs_shape_dict, freeze_params=True)
    elif pathlib.Path(model_file).suffix.lower() == '.tflite':
        import tflite
        tflite_model_buf = open(model_file, "rb").read()
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        inputs_shape_dict = get_input_shape_dict(tflite_model, 'tflite', param)
        inputs_dtype_dict = get_input_dtype_dict(tflite_model, 'tflite')
        tvm_model, tvm_model_params = tvm.relay.frontend.from_tflite(tflite_model, shape_dict=inputs_shape_dict, dtype_dict=inputs_dtype_dict)
    elif pathlib.Path(model_file).suffix.lower() == '.pb':
        import tensorflow as tf
        if os.path.basename(model_file) == 'saved_model.pb':
            from tensorflow.core.protobuf import saved_model_pb2
            from tensorflow2caffe.convert import _remove_non_variable_resources_from_captures, convert_variables_to_constants_v2
            modelpath = model_file.split(os.path.basename(model_file))[0]

            with tf.io.gfile.GFile(model_file, 'rb') as f:
                saved_model = saved_model_pb2.SavedModel()
                saved_model.ParseFromString(tf.compat.as_bytes(f.read()))
                meta_graphs = saved_model.meta_graphs

            if len(meta_graphs) > 1:
                tags = meta_graphs[0].meta_info_def.tags[0]
            else:
                tags = None

            imported = tf.saved_model.load(modelpath, tags=tags)

            all_sigs = imported.signatures.keys()
            valid_sigs = [s for s in all_sigs if not s.startswith("_")]
            concrete_func = imported.signatures[valid_sigs[0]]

            removed_resource_to_placeholder, placeholder_to_resource, graph_captures_copy, func_captures_copy = \
                _remove_non_variable_resources_from_captures(concrete_func)

            inputs = [tensor.name for tensor in concrete_func.inputs if tensor.dtype != tf.dtypes.resource]
            outputs = [tensor.name for tensor in concrete_func.outputs if tensor.dtype != tf.dtypes.resource]

            frozen_func = convert_variables_to_constants_v2(concrete_func, lower_control_flow=True, aggressive_inlining=True)
            graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
        else:
            with tf.compat.v1.gfile.GFile(model_file, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                graph = tf.import_graph_def(graph_def, name="")

        inputs_shape_dict = get_input_shape_dict(graph_def, 'tensorflow', param)
        tvm_model, tvm_model_params = tvm.relay.frontend.from_tensorflow(graph_def, layout=param['layout'], shape=inputs_shape_dict, convert_config={'use_dense':True})
    elif pathlib.Path(model_file).suffix.lower() == '.pt':
        tvm_model, tvm_model_params = tvm.relay.frontend.from_pytorch(model_file, input_infos, custom_convert_map=None, use_parser_friendly_name=False, keep_quantized_weight=False)

    model = Model(tvm_model, tvm_model_params, param)
    model.parse()
    model.convert()
    caffe_net = model.save(caffe_model_path)

    inputs_tensor = list()
    for index, input_name in enumerate(model.inputs):
        inputs_tensor.append(get_input_tensor(param, model.inputs_shape[index], model.inputs_dtype[index], None))

    compare(model, caffe_net, inputs_tensor, param.get('compare', -1))
