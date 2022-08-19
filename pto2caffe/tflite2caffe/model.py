import tflite
import logging
from base import Base
from util import *

from tflite2caffe.op.pad import Pad
from tflite2caffe.op.add import Add
from tflite2caffe.op.mul import Mul
from tflite2caffe.op.sub import Sub
from tflite2caffe.op.relu import ReLU
from tflite2caffe.op.relux import ReLUX
from tflite2caffe.op.prelu import PReLU
from tflite2caffe.op.swish import Swish
from tflite2caffe.op.slice import Slice
from tflite2caffe.op.split import Split
from tflite2caffe.op.concat import Concat
from tflite2caffe.op.bypass import ByPass
from tflite2caffe.op.reshape import Reshape
from tflite2caffe.op.pooling import Pooling
from tflite2caffe.op.sigmoid import Sigmoid
from tflite2caffe.op.softmax import Softmax
from tflite2caffe.op.leakyrelu import LeakyReLU
from tflite2caffe.op.conv import Convolution
from tflite2caffe.op.transpose import Permute
from tflite2caffe.op.reducemax import ReduceMax
from tflite2caffe.op.deconv import Deconvolution
from tflite2caffe.op.reducemean import ReduceMean
from tflite2caffe.op.depthtospace import DepthToSpace
from tflite2caffe.op.stridedslice import StridedSlice
from tflite2caffe.op.fullyconnected import InnerProduct
from tflite2caffe.op.resizenearest import ResizeNearest
from tflite2caffe.op.resizebilinear import ResizeBilinear

from tflite2caffe.op.debug import Debug


from caffe_transform import save_caffe_model
from caffe_transform import make_caffe_input_layer
from tflite2caffe.quantize import Dequantize, isQuantilize


numpy_dtype = [np.float32, np.float16, np.int32, np.uint8, np.int64, 'string', np.bool, np.int16, np.complex64, np.int8, np.float64, np.complex128]

logger = logging.getLogger('TFLite2Caffe')

OpMap = {
    'PAD': Pad,
    'ADD': Add,
    'MUL': Mul,
    'SUB': Sub,
    'RELU': ReLU,
    'CAST': ByPass,
    'PRELU': PReLU,
    'RELU6': ReLUX,
    'SLICE': Slice,
    'SPLIT': Split,
    'QUANTIZE': ByPass,
    'MEAN': ReduceMean,
    'RESHAPE': Reshape,
    'SQUEEZE': Reshape,
    'SOFTMAX': Softmax,
    'LOGISTIC': Sigmoid,
    'HARD_SWISH': Swish,
    'TRANSPOSE': Permute,
    'DEQUANTIZE': ByPass,
    'CONV_2D': Convolution,
    'MAX_POOL_2D': Pooling,
    'REDUCE_MAX': ReduceMax,
    'LEAKY_RELU': LeakyReLU,
    'CONCATENATION': Concat,
    'AVERAGE_POOL_2D': Pooling,
    'STRIDED_SLICE': StridedSlice,
    'DEPTH_TO_SPACE': DepthToSpace,
    'TRANSPOSE_CONV': Deconvolution,
    'FULLY_CONNECTED': InnerProduct,
    'DEPTHWISE_CONV_2D': Convolution,
    'RESIZE_BILINEAR': ResizeBilinear,
    'RESIZE_NEAREST_NEIGHBOR': ResizeNearest,

#    'DIV': Binary,
#   'POW': Binary,
#    'MIRROR_PAD': Pad,
}


def handleFusedActivation(preop):
    if preop.activ_type_code == tflite.ActivationFunctionType.RELU:
        op = ReLU(preop.model, None, 'RELU', preop.index)
    elif preop.activ_type_code == tflite.ActivationFunctionType.RELU_N1_TO_1:
        raise NotImplementedError('ReluN1To1 is not supported.')
    elif preop.activ_type_code == tflite.ActivationFunctionType.RELU6:
        op = ReLUX(preop.model, None, 'RELU6', preop.index)
    elif preop.activ_type_code == tflite.ActivationFunctionType.TANH:
         raise NotImplementedError('Tanh is not supported.')
    elif preop.activ_type_code == tflite.ActivationFunctionType.SIGN_BIT:
         raise NotImplementedError('SignBits is not supported.')
    else:
        return

    for output in preop.outputs:
        op.outputs.append(output)
        op.inputs.append(output)
    op.inputs_buf.append(None)
    op.parse()

    return op


class Model(Base):

    def __init__(self, model:tflite.Model, param):
        super().__init__(model, model.Subgraphs(0))
        self.version = model.Version()

        self.param = param
        self.layout = param['layout']
        self.inputs = list()
        self.inputs_shape = list()
        self.inputs_dtype = list()
        self.inputs_maxval = list()
        self.inputs_minval = list()
        self.inputs_scale = list()
        self.inputs_zeropoint = list()

        self.constant = dict()
        self.indentity = dict()
        self.pad = dict()
        self.operators = list()
        self.unsupport = list()
        self.errorMsg = list()
        self.layers = list()
        self.setInited()


    def parse(self):
        logger.debug("Parsing the TFLite Model...")

        if self.model.SubgraphsLength() > 1:
            raise ValueError('TFLite model include ' + str(self.model.SubgraphsLength()) + ' graphs.')

        print('TFlite Model Input size:')
        for i in range(self.graph.InputsLength()):
            print(self.graph.Tensors(self.graph.Inputs(i)).Name().decode(), ':', list(self.graph.Tensors(self.graph.Inputs(i)).ShapeAsNumpy()))

            tf_dtype = self.graph.Tensors(self.graph.Inputs(i)).Type()
            self.inputs_dtype.append(numpy_dtype[tf_dtype])
            self.inputs_maxval.append(self.graph.Tensors(self.graph.Inputs(i)).Quantization().MaxAsNumpy() if tf_dtype != 0 else None)
            self.inputs_minval.append(self.graph.Tensors(self.graph.Inputs(i)).Quantization().MinAsNumpy() if tf_dtype != 0 else None)
            self.inputs_scale.append(self.graph.Tensors(self.graph.Inputs(i)).Quantization().ScaleAsNumpy() if tf_dtype != 0 else None)
            self.inputs_zeropoint.append(self.graph.Tensors(self.graph.Inputs(i)).Quantization().ZeroPointAsNumpy() if tf_dtype != 0 else None)
            self.inputs_shape.append(shape_map_nhwc2nchw(self.graph.Tensors(self.graph.Inputs(i)).ShapeAsNumpy().tolist()))

        self.param['inputs_shape'] = self.inputs_shape

        # Tensors
        for i in range(self.graph.TensorsLength()):
            type_id = self.graph.Tensors(i).Type()
            buffer_id = self.graph.Tensors(i).Buffer()

            buf = self.model.Buffers(buffer_id).DataAsNumpy()
            shape = self.graph.Tensors(i).ShapeAsNumpy()

            if isinstance(buf, int) and buf == 0:
                self.constant[i] = None
            elif isinstance(buf, np.ndarray):
                nparray = np.frombuffer(buf, dtype=numpy_dtype[type_id]).reshape(shape)
                if self.graph.Tensors(i).Quantization() is not None:
                    quantizedDimmension = self.graph.Tensors(i).Quantization().QuantizedDimension()
                    scale = self.graph.Tensors(i).Quantization().ScaleAsNumpy()
                    zero_point = self.graph.Tensors(i).Quantization().ZeroPointAsNumpy()

                    if isQuantilize(self.graph.Tensors(i).Quantization().ScaleLength(), self.graph.Tensors(i).Quantization().ZeroPointLength()):
                        nparray = Dequantize(nparray, scale, zero_point, quantizedDimmension, np.float32)

                self.constant[i] = nparray
            else:
                raise NotImplementedError

        # Operators
        for index in range(self.graph.OperatorsLength()):
            tf_op = self.graph.Operators(index)
            tf_op_code = self.model.OperatorCodes(tf_op.OpcodeIndex())
            tf_op_name = tflite.opcode2name(tf_op_code.BuiltinCode())

            if tf_op_name in []: #ignore_op
                continue

            if tf_op_name == 'CUSTOM':
                self.unsupport.append(tf_op_code.CustomCode().decode())
                continue

            if tf_op_name not in OpMap:
                self.unsupport.append(tf_op_name)
                continue

            op = OpMap[tf_op_name](self, tf_op, tf_op_name, index)
            op.parse()

            logger.debug(op)
            if op.status.parsed:
                self.operators.append(op)
                if hasattr(op, 'activ_type_code'):
                    act_op = handleFusedActivation(op)
                    self.operators.append(act_op)

        for errorMsg in list(set(self.errorMsg)):
            print(errorMsg)

        if len(self.unsupport) > 0:
            import sys
            errorMsg = 'Error: Operator ' + str(list(set(self.unsupport))) + ' does not Support.\n'
            sys.exit(errorMsg)

        self.setParsed()


    def convert(self):
        logger.debug("Converting the Model...")

        for i in range(self.graph.InputsLength()):
            input_shape = self.graph.Tensors(self.graph.Inputs(i)).ShapeAsNumpy()
            self.layers.append(make_caffe_input_layer(self.graph.Inputs(i), shape_map_nhwc2nchw(input_shape.tolist()), i, self.param))

        for op in self.operators:
            logger.debug(op)
            self.layers.extend(op.convert())

        self.setConverted()


    def save(self, caffe_model_path):
        save_caffe_model(caffe_model_path, self.layers)
