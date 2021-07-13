import tflite
import logging

from caffe_transform import save_caffe_model
from caffe_transform import make_caffe_input_layer
from tflite2caffe.op.operator import Operator

from tflite2caffe.op.pad import Pad
from tflite2caffe.op.binary import Binary
from tflite2caffe.op.resize import Resize 
from tflite2caffe.op.conv import Convolution
from tflite2caffe.op.concat import Concat
from tflite2caffe.op.activation import Activation

from base import Base

logger = logging.getLogger('TFlite2caffe')

OpMap = { 
    'CONV_2D': Convolution,
    'LEAKY_RELU': Activation,
    'PAD': Pad,
    'ADD': Binary,
    'RESIZE_NEAREST_NEIGHBOR': Resize,
    'CONCATENATION': Concat
#    'DEPTHWISE_CONV_2D': Depthwise_conv2d,
#    'MEAN': Mean,
#    'AVERAGE_POOL_2D': AvgPool2d,
#    'MAX_POOL_2D': MaxPool2d,
#    'RESHAPE': Reshape,
#    'SOFTMAX': Softmax,
#    'MUL': Mul,
#    'FULLY_CONNECTED': Fully_connected
}

def opFactory(tfmodel, graph, tf_op, index, legacys):
    tf_op_code = tfmodel.OperatorCodes(tf_op.OpcodeIndex())
    tf_op_name = tflite.opcode2name(tf_op_code.BuiltinCode())

    return OpMap[tf_op_name](tfmodel, graph, tf_op, tf_op_code, index, legacys)


class Model(Base):
    def __init__(self, model:tflite.Model, param):
        super().__init__(model)
        self.tfmodel = model
        self.param = param
        self.operators = []
        self.setInited()
        self.layers = []


    def parse(self):
        logger.debug("Parsing the Model...")
        if self.tfmodel.SubgraphsLength() > 1:
            raise ValueError('TFlite model include ' + str(self.tfmodel.SubgraphsLength()) + 'graph.')

        self.graph = self.tfmodel.Subgraphs(0)

        legacys = []
        for index in range(self.graph.OperatorsLength()):
            tf_op = self.graph.Operators(index)
            op = opFactory(self.tfmodel, self.graph, tf_op, index, legacys)
            op.parse()
            if op.status.parsed:
                self.operators.append(op)
            else:
                legacys.append(op)

        self.setParsed()


    def convert(self):
        self.parse()
        logger.debug("Converting the Model...")

        self.layers.append(make_caffe_input_layer(self.graph.Inputs(0), self.param))
        for op in self.operators:
            print(op)
            layer = op.convert()
            self.layers.append(layer)
        self.setConverted()


    def save(self, caffe_name, caffe_path):
        save_caffe_model(caffe_name, caffe_path, self.layers)
