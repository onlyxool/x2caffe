import flatbuffers
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

def OutputsOffset(subgraph, j):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(subgraph._tab.Offset(8))
    if o != 0:
        a = subgraph._tab.Vector(o)
        return a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4)
    return 0


#参考了https://github.com/raymond-li/tflite_tensor_outputter/blob/master/tflite_tensor_outputter.py
#调整output到指定idx
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


def get_output(model, input_tensor, op, index):
    model = buffer_change_output_tensor_to(model, op.outputs[index])
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Model Run
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)

    return output.transpose(0, 3, 1, 2) if len(output.shape) == 4 else output
