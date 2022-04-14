import sys
import tflite
import logging
from dump import Dump
from base import Base
from util import *

from tflite2caffe.op.pad import Pad
from tflite2caffe.op.relu import ReLU
from tflite2caffe.op.relux import ReLUX
from tflite2caffe.op.prelu import PReLU
from tflite2caffe.op.swish import Swish
from tflite2caffe.op.split import Slice #TODO
from tflite2caffe.op.binary import Binary
from tflite2caffe.op.reduce import Reduce
from tflite2caffe.op.resize import Resize
from tflite2caffe.op.concat import Concat
from tflite2caffe.op.reshape import Reshape
from tflite2caffe.op.pooling import Pooling
from tflite2caffe.op.sigmoid import Sigmoid
from tflite2caffe.op.softmax import Softmax
from tflite2caffe.op.conv import Convolution
from tflite2caffe.op.quantize import Quantize
from tflite2caffe.op.transpose import Permute
from tflite2caffe.op.deconv import Deconvolution
from tflite2caffe.op.fullyconnected import InnerProduct

from caffe_transform import save_caffe_model
from caffe_transform import make_caffe_input_layer


logger = logging.getLogger('TFLite2Caffe')

OpMap = {
    'PAD': Pad,
    'RELU': ReLU,
    'ADD': Binary,
    'MUL': Binary,
    'SUB': Binary,
    'MEAN': Reduce,
    'PRELU': PReLU,
    'SPLIT': Slice,
    'RESHAPE': Reshape,
    'SQUEEZE': Reshape,
    'SOFTMAX': Softmax,
    'LEAKY_RELU': ReLU,
    'LOGISTIC': Sigmoid,
    'HARD_SWISH': Swish,
    'QUANTIZE': Quantize,
    'TRANSPOSE': Permute,
    'REDUCE_MAX': Reduce,
    'CONV_2D': Convolution,
    'DEQUANTIZE': Quantize,
    'MAX_POOL_2D': Pooling,
#    'STRIDED_SLICE': Slice,
    'CONCATENATION': Concat,
    'RESIZE_BILINEAR': Resize,
    'AVERAGE_POOL_2D': Pooling,
    'TRANSPOSE_CONV': Deconvolution,
    'FULLY_CONNECTED': InnerProduct,
    'DEPTHWISE_CONV_2D': Convolution,
    'RESIZE_NEAREST_NEIGHBOR': Resize,
#    'DIV': Binary,
#    'MIRROR_PAD': Pad,
}

ignore_op = ['CUSTOM']


def handleFusedActivation(preop):
    if preop.activ_type_code == tflite.ActivationFunctionType.RELU:
        op = ReLU(preop.model, None, tflite.BuiltinOperator.RELU, preop.index)
    elif preop.activ_type_code == tflite.ActivationFunctionType.RELU_N1_TO_1:
        raise NotImplementedError('ReluN1To1 is not supported.')
    elif preop.activ_type_code == tflite.ActivationFunctionType.RELU6:
        op = ReLUX(preop.model, None, tflite.BuiltinOperator.RELU6, preop.index)
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
        self.inputs_shape = list()
        self.param = param
        self.operators = []
        self.layers = []
        self.legacys = []
        self.setInited()


    def parse(self):
        logger.debug("Parsing the TFLite Model...")

        if self.model.SubgraphsLength() > 1:
            raise ValueError('TFLite model include ' + str(self.model.SubgraphsLength()) + ' graphs.')

        print('TFlite Model Input size:')
        for i in range(self.graph.InputsLength()):
            print(self.graph.Inputs(i), ':', list(self.graph.Tensors(self.graph.Inputs(i)).ShapeAsNumpy()))
            self.inputs_shape.append(list(self.graph.Tensors(self.graph.Inputs(i)).ShapeAsNumpy()))
        self.param['inputs_shape'] = self.inputs_shape

        for index in range(self.graph.OperatorsLength()):
            tf_op = self.graph.Operators(index)
            tf_op_code = self.model.OperatorCodes(tf_op.OpcodeIndex())
            tf_op_name = tflite.opcode2name(tf_op_code.BuiltinCode())

            if tf_op_name in ignore_op:
                continue
            if tf_op_name not in OpMap:
                errorMsg = 'Error: Operator [' + tf_op_name + '] does not Support.\n'
                sys.exit(errorMsg)

            op = OpMap[tf_op_name](self, tf_op, tf_op_code.BuiltinCode(), index)
            op.parse()

            logger.debug(op)
            if op.status.parsed:
                self.operators.append(op)
                if hasattr(op, 'activ_type_code'):
                    act_op = handleFusedActivation(op)
                    self.operators.append(act_op)
            else:
                self.legacys.append(op)

        self.setParsed()


    def convert(self):
        logger.debug("Converting the Model...")

        for i in range(self.graph.InputsLength()):
            input_shape = self.graph.Tensors(self.graph.Inputs(i)).ShapeAsNumpy()
            self.layers.append(make_caffe_input_layer(self.graph.Inputs(i), shape_map_nhwc2nchw(input_shape), i, self.param))

        for op in self.operators:
            logger.debug(op)
            layers = op.convert()
            for layer in layers:
                self.layers.append(layer)

        self.setConverted()


    def save(self, caffe_path):
        save_caffe_model(caffe_path, self.layers)


    def dump(self, model_byte, model_name, input_tensor, dump_level=-1):
        dump = Dump('TFLite', model_byte, model_name, input_tensor, self.param, dump_level)
        from progress_bar import ProgressBar
        progressBar = ProgressBar(len(self.operators), 0, "TFLite dump processing")
        for i, op in enumerate(self.operators):
            dump.operator(op)
            progressBar.setValue(i)
        progressBar.onCancel()
