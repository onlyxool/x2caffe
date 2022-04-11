import copy
import logging
import numpy as np
import flatbuffers
import tensorflow as tf
from tflite2caffe.quantize import Dequantize, Quantize, isQuantilize, checkQuantilize
from tensorflow.lite.python import schema_py_generated as schema_fb

logger = logging.getLogger('tflite2caffe')


def OutputsOffset(subgraph, j):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(subgraph._tab.Offset(8))
    if o != 0:
        a = subgraph._tab.Vector(o)
        return a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4)
    return 0


# Ref: https://github.com/raymond-li/tflite_tensor_outputter/blob/master/tflite_tensor_outputter.py
# Modify model output to specific blob
def buffer_change_output_tensor_to(model_buffer, new_tensor_i):
    fb_model_root = schema_fb.Model.GetRootAsModel(model_buffer, 0)
    output_tensor_index_offset = OutputsOffset(fb_model_root.Subgraphs(0), 0)

    # Flatbuffer scalars are stored in little-endian.
    new_tensor_i_bytes = bytes([
        new_tensor_i & 0x000000FF, \
        (new_tensor_i & 0x0000FF00) >> 8, \
        (new_tensor_i & 0x00FF0000) >> 16, \
        (new_tensor_i & 0xFF000000) >> 24 \
    ])

    # Replace the 4 bytes corresponding to the first output tensor index
    return model_buffer[:output_tensor_index_offset] + new_tensor_i_bytes + model_buffer[output_tensor_index_offset + 4:]


def get_output(model, input_tensor, blob_name):
    if isinstance(blob_name, str):
        try:
            blob_name = int(blob_name.split('_')[0])
        except:
            return None

    model = buffer_change_output_tensor_to(model, blob_name)

    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Quantize Input
    for detail in input_details:
        scales = detail['quantization_parameters']['scales']
        zero_points = detail['quantization_parameters']['zero_points']
        quantized_dimension = detail['quantization_parameters']['quantized_dimension']
        dtype = detail['dtype']
        if isQuantilize(len(scales), len(zero_points)) and checkQuantilize(input_tensor, scales, zero_points, dtype, quantized_dimension):
            input_tensor = Quantize(input_tensor, scales, zero_points, quantized_dimension, dtype)

        interpreter.set_tensor(detail['index'], input_tensor)

    # Model Inference
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    # Dequantize Output
    for detail in output_details:
        if detail['index'] == blob_name:
            scales = detail['quantization_parameters']['scales']
            zero_points = detail['quantization_parameters']['zero_points']
            quantized_dimension = detail['quantization_parameters']['quantized_dimension']
            dtype = detail['dtype']
            if isQuantilize(len(scales), len(zero_points)):
                print('Dequantize Blob', dtype, 'to', np.float32)
                output = Dequantize(output, scales, zero_points, quantized_dimension, dtype)
    return output.transpose(0, 3, 1, 2) if len(output.shape) == 4 else output
