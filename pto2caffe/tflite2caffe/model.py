import sys
import tflite
import logging
import numpy as np
import tensorflow as tf
from base_Model import BaseModel

from tflite2caffe.op.add import Add
from tflite2caffe.op.mul import Mul
from tflite2caffe.op.pad import Pad
from tflite2caffe.op.sub import Sub
from tflite2caffe.op.pack import Pack
from tflite2caffe.op.relu import ReLU
from tflite2caffe.op.prelu import PReLU
from tflite2caffe.op.relux import ReLUX
from tflite2caffe.op.shape import Shape
from tflite2caffe.op.slice import Slice
from tflite2caffe.op.split import Split
from tflite2caffe.op.swish import Swish
from tflite2caffe.op.bypass import ByPass
from tflite2caffe.op.concat import Concat
from tflite2caffe.op.pooling import Pooling
from tflite2caffe.op.reshape import Reshape
from tflite2caffe.op.sigmoid import Sigmoid
from tflite2caffe.op.softmax import Softmax
from tflite2caffe.op.conv import Convolution
from tflite2caffe.op.transpose import Permute
from tflite2caffe.op.leakyrelu import LeakyReLU
from tflite2caffe.op.reducemax import ReduceMax
from tflite2caffe.op.deconv import Deconvolution
from tflite2caffe.op.reducemean import ReduceMean
from tflite2caffe.op.depthtospace import DepthToSpace
from tflite2caffe.op.stridedslice import StridedSlice
from tflite2caffe.op.resizenearest import ResizeNearest
from tflite2caffe.op.fullyconnected import InnerProduct
from tflite2caffe.op.resizebilinear import ResizeBilinear

from util import shape_map_nhwc2nchw
from caffe_transform import save_caffe_model
from caffe_transform import make_caffe_input_layer


numpy_dtype = [np.float32, np.float16, np.int32, np.uint8, np.int64, 'string', np.bool, np.int16, np.complex64, np.int8, np.float64, np.complex128]

tf_dtype_map = [tf.float32, tf.float16, tf.qint32, tf.quint8, tf.int64, 'string', tf.bool, tf.qint16, tf.complex64, tf.qint8, tf.float64, tf.complex128]

logger = logging.getLogger('TFLite2Caffe')

OpMap = {
    'ADD': Add,
    'MUL': Mul,
    'PAD': Pad,
    'SUB': Sub,
    'PACK': Pack,
    'RELU': ReLU,
    'CAST': ByPass,
    'PRELU': PReLU,
    'RELU6': ReLUX,
    'SHAPE': Shape,
    'SLICE': Slice,
    'SPLIT': Split,
    'MEAN': ReduceMean,
    'RESHAPE': Reshape,
    'SQUEEZE': Reshape,
    'SOFTMAX': Softmax,
    'QUANTIZE': ByPass,
    'LOGISTIC': Sigmoid,
    'HARD_SWISH': Swish,
    'TRANSPOSE': Permute,
    'DEQUANTIZE': ByPass,
    'CONV_2D': Convolution,
    'MAX_POOL_2D': Pooling,
    'LEAKY_RELU': LeakyReLU,
    'REDUCE_MAX': ReduceMax,
    'CONCATENATION': Concat,
    'AVERAGE_POOL_2D': Pooling,
    'STRIDED_SLICE': StridedSlice,
    'DEPTH_TO_SPACE': DepthToSpace,
    'TRANSPOSE_CONV': Deconvolution,
    'FULLY_CONNECTED': InnerProduct,
    'DEPTHWISE_CONV_2D': Convolution,
    'RESIZE_BILINEAR': ResizeBilinear,
    'RESIZE_NEAREST_NEIGHBOR': ResizeNearest,
}

ActOpMap = [
    tflite.ActivationFunctionType.RELU,
    tflite.ActivationFunctionType.RELU6,
]

UnSupprotActOp = {
    tflite.ActivationFunctionType.TANH: 'TANH',
    tflite.ActivationFunctionType.SIGN_BIT: 'SIGN_BIT',
    tflite.ActivationFunctionType.RELU_N1_TO_1: 'RELU_N1_TO_1',
}

def handleFusedActivation(op):
    if op.activ_type_code == tflite.ActivationFunctionType.RELU:
        actop = ReLU(op.model, None, 'RELU', op.index)
    elif op.activ_type_code == tflite.ActivationFunctionType.RELU6:
        actop = ReLUX(op.model, None, 'RELU6', op.index)
    actop.isactivate = True

    for output in op.outputs:
        actop.outputs.append(output)
        actop.inputs.append(output)
    actop.inputs_buf.append(None)

    actop.parse()

    return actop


class Model(BaseModel):

    def __init__(self, model:tflite.Model, param, model_byte):
        super().__init__(model, model.Subgraphs(0), param)
        self.version = model.Version()
        self.model_byte = model_byte
        self.setInited()


    def parse(self):
        logger.debug("Parsing the TFLite Model...")

        if self.model.SubgraphsLength() > 1:
            errorMsg = '\nError: TFLite model include more than one graphs: ' + str(self.model.SubgraphsLength()) + '\n'
            sys.exit(errorMsg)

        print('TFLite Model Input size: (graph version=%d)' %self.version)
        for i in range(self.graph.InputsLength()):
            print(self.graph.Tensors(self.graph.Inputs(i)).Name().decode(), ':', list(self.graph.Tensors(self.graph.Inputs(i)).ShapeAsNumpy()))
            self.inputs.append(self.graph.Inputs(i))
            self.inputs_dtype.append(numpy_dtype[self.graph.Tensors(self.graph.Inputs(i)).Type()])
            input_shape = self.graph.Tensors(self.graph.Inputs(i)).ShapeAsNumpy().tolist()
            self.inputs_shape.append(shape_map_nhwc2nchw(input_shape) if self.layout == 'NHWC' else input_shape)
            self.inputs_quantization_parameter.append(self.get_tensor_quantization_parameter(self.graph.Inputs(i)))

        # Tensors
        for i in range(self.graph.TensorsLength()):
            quantization_parameter = self.get_tensor_quantization_parameter(i)

            raw = self.model.Buffers(self.graph.Tensors(i).Buffer()).DataAsNumpy()

            if isinstance(raw, int) and raw == 0:
                self.constant[i] = None
            elif isinstance(raw, np.ndarray):
                if isinstance(self.graph.Tensors(i).ShapeAsNumpy(), int) and self.graph.Tensors(i).ShapeAsNumpy() == 0:
                    constant = np.frombuffer(raw, dtype=numpy_dtype[self.graph.Tensors(i).Type()])[0]
                else:
                    constant = np.frombuffer(raw, dtype=numpy_dtype[self.graph.Tensors(i).Type()]).reshape(self.graph.Tensors(i).ShapeAsNumpy())
                self.constant[i] = self.dequantize(constant, i)
            else:
                sys.exit('Can\'t Read Model Buffer.')

        # Operators
        for index in range(self.graph.OperatorsLength()):
            tf_op = self.graph.Operators(index)
            tf_op_code = self.model.OperatorCodes(tf_op.OpcodeIndex())
            tf_op_name = tflite.opcode2name(tf_op_code.BuiltinCode())

            if tf_op_name in (): #ignore_op
                continue

            if tf_op_name == 'CUSTOM':
                if tf_op_code.CustomCode().decode() in ('TFLite_Detection_PostProcess',):
                    continue
                else:
                    self.unsupport.append(tf_op_code.CustomCode().decode())
                    continue

            if tf_op_name not in OpMap:
                self.unsupport.append(tf_op_name)
                continue

            op = OpMap[tf_op_name](self, tf_op, tf_op_name, index)
            op.parse()

            if op.status.parsed:
                self.operators.append(op)

            # FusedActivation
            if op.activ_type_code in ActOpMap:
                self.operators.append(handleFusedActivation(op))
            elif op.activ_type_code in UnSupprotActOp:
                self.unsupport.append(UnSupprotActOp[op.activ_type_code])


        for errorMsg in list(set(self.errorMsg)):
            print(errorMsg)

        if len(self.unsupport) > 0:
            errorMsg = 'Error: Operator ' + str(list(set(self.unsupport))) + ' does not Support.\n'
            sys.exit(errorMsg)

        self.setParsed()


    def convert(self):
        logger.debug("Converting the Model...")

        for index, input_name in enumerate(self.inputs):
            self.layers.append(make_caffe_input_layer(input_name, self.inputs_shape[index], index, self.param))

        for op in self.operators:
            self.layers.extend(op.convert())

        self.setConverted()


    def save(self, caffe_model_path):
        return save_caffe_model(caffe_model_path, self.layers)


    def quantize(self, tensor, index):
        quantization_parameter = self.get_tensor_quantization_parameter(index)

        if quantization_parameter is not None:
            dtype = quantization_parameter['dtype']
            scale = quantization_parameter['scale']
            min_range = quantization_parameter['minval']
            max_range = quantization_parameter['maxval']
            zero_point = quantization_parameter['zero_point']
            axis = quantization_parameter['quantized_dimension']
            return (tensor/scale + zero_point).astype(np.uint8)
#            return tf.raw_ops.QuantizeV2(input=tensor, min_range=min_range, max_range=max_range, T=dtype, mode='MIN_COMBINED',
#                    round_mode='HALF_AWAY_FROM_ZERO', narrow_range=False, axis=-1, ensure_minimum_range=0.01, name=None)[0].numpy()
        else:
            return tensor


    def dequantize(self, tensor, index):
        quantization_parameter = self.get_tensor_quantization_parameter(index)

        if quantization_parameter is not None:
            dtype = quantization_parameter['dtype']
            scale = quantization_parameter['scale']
            zero_point = quantization_parameter['zero_point']
            min_range = quantization_parameter['minval']
            max_range = quantization_parameter['maxval']
            axis = quantization_parameter['quantized_dimension']
#            return tf.raw_ops.Dequantize(input=tf.constant(tensor, dtype=dtype), min_range=min_range, max_range=max_range,
#                            mode='MIN_COMBINED',  narrow_range=False, axis=-1, dtype=tf.dtypes.float32, name=None).numpy()
            return (tensor.astype(np.float32) - zero_point) * scale
        else:
            return tensor


    def get_tensor_quantization_parameter(self, index):
        dtype = self.graph.Tensors(index).Type()
        if dtype == 0:
            return None

        maxval = self.graph.Tensors(index).Quantization().MaxAsNumpy()
        minval = self.graph.Tensors(index).Quantization().MinAsNumpy()
        scale = self.graph.Tensors(index).Quantization().ScaleAsNumpy()
        zero_point = self.graph.Tensors(index).Quantization().ZeroPointAsNumpy()
        quantized_dimension = self.graph.Tensors(index).Quantization().QuantizedDimension()

        if isinstance(scale, np.ndarray) and isinstance(zero_point, np.ndarray):
            return {'dtype': tf_dtype_map[dtype],
                    'maxval':maxval, 'minval': minval,
                    'scale': scale, 'zero_point': zero_point,
                    'quantized_dimension': quantized_dimension}
        else:
            return None


    def forward(self, output_name, inputs_tensor):
        if output_name.split('_')[0].isdigit():
            output_name = int(output_name.split('_')[0])
        else:
            return None

        def OutputsOffset(subgraph, j):
            import flatbuffers
            o = flatbuffers.number_types.UOffsetTFlags.py_type(subgraph._tab.Offset(8))
            if o != 0:
                a = subgraph._tab.Vector(o)
                return a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4)
            return 0

        from tensorflow.lite.python import schema_py_generated as schema_fb
        fb_model_root = schema_fb.Model.GetRootAsModel(self.model_byte, 0)
        output_tensor_index_offset = OutputsOffset(fb_model_root.Subgraphs(0), 0)

        # Flatbuffer scalars are stored in little-endian.
        new_tensor_i_bytes = bytes([
             output_name & 0x000000FF, \
            (output_name & 0x0000FF00) >> 8, \
            (output_name & 0x00FF0000) >> 16, \
            (output_name & 0xFF000000) >> 24 \
        ])
        # Replace the 4 bytes corresponding to the first output tensor index
        model = self.model_byte[:output_tensor_index_offset] + new_tensor_i_bytes + self.model_byte[output_tensor_index_offset + 4:]


        interpreter = tf.lite.Interpreter(model_content=model, num_threads=None)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Quantize Input
        for index, input_info in enumerate(input_details):
            input_tensor = self.quantize(inputs_tensor[index], input_info['index'])
            input_tensor = input_tensor.transpose(0, 2, 3, 1) if self.layout == 'NHWC' and input_info['shape'].size == 4 else input_tensor
            interpreter.set_tensor(input_info['index'], input_tensor)

        # Model Inference
        interpreter.invoke()
        output_tensor = interpreter.get_tensor(output_details[0]["index"])

        # Dequantize Output
        for output_info in output_details:
            output_tensor = self.dequantize(output_tensor, output_info['index'])

        return output_tensor.transpose(0, 3, 1, 2) if self.layout == 'NHWC' and len(output_tensor.shape) == 4 else output_tensor
