import onnx
import onnxruntime
import numpy as np
from onnx import helper


def shape_proto2list(proto_shape):
    list_shape = []
    for dim in proto_shape.dim:
        list_shape.append(dim.dim_value)

    return list_shape


def onnx_run(model, input_tensor):
    onnxruntime.set_default_logger_severity(4)
    if onnxruntime.get_device() == 'GPU':
        onnx_session = onnxruntime.InferenceSession(model.SerializeToString(), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    if onnxruntime.get_device() == 'CPU':
        onnx_session = onnxruntime.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])


    input_name = []
    for node in onnx_session.get_inputs():
        input_name.append(node.name)

    output_name = []
    for node in onnx_session.get_outputs():
        output_name.append(node.name)

    input_feed = {}
    for name in input_name:
        input_feed[name] = input_tensor

    return onnx_session.run(output_name, input_feed=input_feed)


def get_output(model, input_tensor, blob_name):
    for value_info in model.graph.value_info:
        if value_info.name == blob_name:
            # insert output
            shape_list = shape_proto2list(value_info.type.tensor_type.shape)
            blob_info = helper.make_tensor_value_info(blob_name, onnx.TensorProto.FLOAT, shape_list)
            model.graph.output.insert(0, blob_info)
            output = onnx_run(model, input_tensor)
            return np.array(output[0])
        else:
            for index, output in enumerate(model.graph.output):
                if blob_name == output.name:
                    outputs = onnx_run(model, input_tensor)
                    return np.array(outputs[index])
