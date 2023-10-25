import copy
import ctypes
import numpy as np
from ctypes import *

pnnx = ctypes.CDLL('pytorch2caffe/libpnnx.so', ctypes.RTLD_GLOBAL)

# Model
parse_torch_script = pnnx.parse
parse_torch_script.argtypes = [c_char_p, c_char_p]
parse_torch_script.restype = c_uint

model_forward = pnnx.model_forward
model_forward.argtypes = [c_char_p, c_char_p]
model_forward.restype = c_int

get_ops_output = pnnx.get_ops_output
get_ops_output.argtypes = [c_char_p, c_char_p]
get_ops_output.restype = c_int

# Operator
get_ops_len = pnnx.get_ops_len
get_ops_len.argtypes = []
get_ops_len.restype = c_uint

get_ops_type = pnnx.get_ops_type
get_ops_type.argtypes = [c_uint]
get_ops_type.restype = c_char_p

get_ops_name = pnnx.get_ops_name
get_ops_name.argtypes = [c_uint]
get_ops_name.restype = c_char_p

# Input Operand
get_inputs_len = pnnx.get_inputs_len
get_inputs_len.argtypes = [c_uint]
get_inputs_len.restype = c_uint

get_input_name = pnnx.get_input_name
get_input_name.argtypes = [c_uint, c_uint]
get_input_name.restype = c_char_p

get_ops_input_shape = pnnx.get_ops_input_shape
get_ops_input_shape.argtype = [c_uint, c_uint]
get_ops_input_shape.restype = c_char_p

# Output Operand
get_outputs_len = pnnx.get_outputs_len
get_outputs_len.argtype = [c_uint]
get_outputs_len.restype = c_uint

get_output_name = pnnx.get_output_name
get_output_name.argtype = [c_uint, c_uint]
get_output_name.restype = c_char_p

get_ops_output_shape = pnnx.get_ops_output_shape
get_ops_output_shape.argtype = [c_uint, c_uint]
get_ops_output_shape.restype = c_char_p

# Attribute
get_ops_attrs_len = pnnx.get_ops_attrs_len
get_ops_attrs_len.argtype = [c_uint]
get_ops_attrs_len.restype = c_uint

get_ops_attrs_names = pnnx.get_ops_attrs_names
get_ops_attrs_names.argtype = [c_uint]
get_ops_attrs_names.restype = c_char_p

get_ops_attr_type = pnnx.get_ops_attr_type
get_ops_attr_type.argtype = [c_uint, c_char_p]
get_ops_attr_type.restype = c_uint

get_ops_attr_shape = pnnx.get_ops_attr_shape
get_ops_attr_shape.argtype = [c_uint, c_char_p]
get_ops_attr_shape.restype = c_char_p

get_ops_attr = pnnx.get_ops_attr
get_ops_attr.argtype = [c_uint, c_char_p, c_char_p]
get_ops_attr.restype = c_uint

get_ops_attr_data_size = pnnx.get_ops_attr_data_size
get_ops_attr_data_size.argtype = [c_uint, c_char_p]
get_ops_attr_data_size.restype = c_uint

# Parameter
get_ops_param = pnnx.get_ops_param
get_ops_param.argtype = [c_uint]
get_ops_param.restype = c_char_p


#0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8
AttrType = [None, np.float32, np.float64, np.float16, np.int32, np.int64, np.int16, np.int8, np.uint8]

def str2shape(dims_str):
    shape = []
    for dim in dims_str.split(','):
        shape.append(int(dim))
    return shape

def shape2str(shape):
    shape_str = str()
    for dim in shape:
        shape_str = shape_str + str(dim) + ','

    return shape_str

class Pnnx():

    def __init__(self, pytorch_file, inputs_shape):
        inputs_shape_str = str()
        for input_shape in inputs_shape:
            for dim in input_shape:
                inputs_shape_str = inputs_shape_str + str(dim) + ','
            inputs_shape_str += '|'

        parse_torch_script(pytorch_file.encode(), inputs_shape_str.encode())


    def model_forward(self, input_tensor):
        model_forward((input_tensor.data.tobytes()), shape2str(input_tensor.shape).encode())


    def get_ops_output(self, layer_name, size):
        data_size = size*4
        output_buf = (c_char * data_size)()

        if size == get_ops_output(layer_name.encode(), output_buf):
            return np.frombuffer(output_buf, dtype=AttrType[1])


    def get_ops_name(self, ops_no):
        return get_ops_name(ops_no).decode()


    def get_ops_type(self):
        ops_type = []
        for i in range(get_ops_len()):
            ops_type.append(get_ops_type(i).decode())

        return ops_type


    def get_ops_attrs_names(self, ops_no):
        return get_ops_attrs_names(ops_no).decode().split(',')


    def get_ops_attr_type(self, ops_no, name):
        return get_ops_attr_type(ops_no, name.encode())


    def get_ops_inputs(self, ops_no):
        inputs_name = []
        for i in range(get_inputs_len(ops_no)):
            inputs_name.append(get_input_name(ops_no, i).decode())

        attrs_name = self.get_ops_attrs_names(ops_no)
        if attrs_name[0] != '': 
            inputs_name.extend(attrs_name)

        return inputs_name


    def get_ops_outputs(self, ops_no):
        outputs_name = []
        for i in range(get_outputs_len(ops_no)):
            outputs_name.append(get_output_name(ops_no, i).decode())

        return outputs_name


    def get_ops_input_shape(self, ops_no, operand_no, name):
        dims_str = get_ops_input_shape(ops_no, operand_no).decode()
        if dims_str != '':
            return str2shape(dims_str)

        dims_str = get_ops_attr_shape(ops_no, name.encode()).decode()
        if dims_str != '':
            return str2shape(dims_str)


    def get_ops_output_shape(self, ops_no, operand_no):
        dims_str = get_ops_output_shape(ops_no, operand_no).decode()
        if dims_str != '':
            return str2shape(dims_str)


    def get_ops_attr(self, ops_no, name):
        size = get_ops_attr_data_size(ops_no, name.encode())
        type_no = get_ops_attr_type(ops_no, name.encode())

        if size != 0 and type_no != 0:
            buf = (c_char * size)()
            get_ops_attr(ops_no, name.encode(), buf)
            return np.frombuffer(buf, dtype=AttrType[type_no])


    def get_ops_param(self, ops_no):
        return get_ops_param(ops_no).decode()


    def get_ops_attr_shape(self, ops_no, attr_name):
        shape = []
        dims_str = get_ops_attr_shape(ops_no, attr_name.encode()).decode().split(',')
        for dim_str in dims_str:
            shape.append(int(dim_str))

        return shape
