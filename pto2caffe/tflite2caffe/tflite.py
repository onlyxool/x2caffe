import copy
import logging
import numpy as np
import flatbuffers
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

logger = logging.getLogger('TFLite')


def isQuantilize(detail):
    scales = detail['quantization_parameters']['scales']
    zero_points = detail['quantization_parameters']['zero_points']
    if scales.size != 0 and zero_points.size != 0:
        return True
    else:
        return False


def quantize(tensor, detail): #TODO
    quantization = detail['quantization_parameters']
    if quantization == (0.0, 0):
        return tensor

    scales = quantization['scales']
    zero_points = quantization['zero_points']

    tensor_scaled = np.divide(tensor, scales)
    tensor_quant = np.add(tensor_scaled, zero_points)

    return tensor_quant.astype(detail['dtype'])


def dequantize(tensor, detail): #TODO
    quantization = detail['quantization_parameters']
    if quantization == (0.0, 0):
        return tensor

    logger.debug("Dequantizing", detail['name'], detail['index'])
#    print(quantization)
    scales = quantization['scales']
    zero_points = quantization['zero_points']
    if quantization['quantized_dimension'] == 0:
        tensor_int32 = tensor.astype('int32')
        tensor_shiftted = np.subtract(tensor_int32, zero_points)
        tensor_fp32 = np.multiply(tensor_shiftted.astype('float32'), scales)
    else:
        raise NotImplementedError(type(quantization['quantized_dimension']))

    return tensor_fp32


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

#    signature = interpreter.get_signature_list()
#    if not signature:
#        encode = interpreter.get_signature_runner('encode')
#        decode = interpreter.get_signature_runner('decode')
#
#        input = tf.constant([1, 2, 3], dtype=tf.float32)
#        print('Input:', input)
#        encoded = encode(x=input)
#        print('Encoded result:', encoded)
#        decoded = decode(x=encoded['encoded_result'])
#        print('Decoded result:', decoded)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()


    for detail in input_details:
#        print(input_tensor)
#        print(detail)
        if isQuantilize(detail):
            input_tensor = quantize(input_tensor, detail) #TODO
#        print(input_tensor, 'oooooooo')
#input_tensor = tf.convert_to_tensor(input_tensor)
        interpreter.set_tensor(detail['index'], input_tensor)

#input_tensor = input_tensor.astype(input_details[0]['dtype'])

    # Model Run
    interpreter.invoke()
    output_index = interpreter.get_output_details()[0]["index"]
    output = interpreter.get_tensor(output_index)

    # Dequantize
    output_details = interpreter.get_output_details()
    for detail in output_details:
        if detail['index'] == blob_name:
            if isQuantilize(detail):
                print('Dequantize output:', str(detail['dtype']), ' to float32')
#                print(output)
                output = dequantize(output, detail)

    return output.transpose(0, 3, 1, 2) if len(output.shape) == 4 else output
