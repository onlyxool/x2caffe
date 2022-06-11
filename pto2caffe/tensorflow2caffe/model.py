import sys
import logging
from base import Base
import tensorflow as tf

from tensorflow2caffe.op.pad import Pad
from tensorflow2caffe.op.add import Add
from tensorflow2caffe.op.sub import Sub
from tensorflow2caffe.op.mul import Mul
from tensorflow2caffe.op.pool import Pool
from tensorflow2caffe.op.mean import Mean
from tensorflow2caffe.op.relu import ReLU
from tensorflow2caffe.op.relux import ReLUX
from tensorflow2caffe.op.resize import Resize
from tensorflow2caffe.op.concat import Concat
from tensorflow2caffe.op.matmul import MatMul
from tensorflow2caffe.op.reshape import Reshape
from tensorflow2caffe.op.softmax import Softmax
from tensorflow2caffe.op.biasadd import BiasAdd
from tensorflow2caffe.op.placeholder import Input
from tensorflow2caffe.op.conv2d import Convolution
from tensorflow2caffe.op.batchnorm import BatchNorm
from tensorflow2caffe.op.leakyrelu import LeakyRelu
from tensorflow2caffe.op.spacetodepth import SpaceToDepth
from tensorflow2caffe.op.depthwise import ConvolutionDepthwise

#from tensorflow2caffe.op.debug import Debug

from caffe_transform import save_caffe_model
from caffe_transform import make_caffe_input_layer
from util import shape_map_nhwc2nchw


def getDynamicInputShape(graph):
    import numpy as np
    dynamic_input = dict()
    for op in graph.get_operations():
        if op.type == 'Placeholder':
            inputs_shape = np.array([dim.size for dim in op.get_attr('shape').dim])
            dynamic_input[op.outputs[0].name] = (inputs_shape == np.array([-1, -1, -1, -1]))

    return dynamic_input


def shape_inference(frozen_func, graph, dynamic_input, param):
    if param['input_shape'] is not None:
        for input_name, dynamic_dim in dynamic_input.items():
            frozen_func.graph.get_tensor_by_name(input_name).set_shape(param['input_shape'])

        with tf.Graph().as_default() as inferred_graph:
            tf.import_graph_def(frozen_func.graph.as_graph_def(add_shapes=True), name="")

        return inferred_graph
    else:
        return graph


logger = logging.getLogger('TensorFlow2Caffe')

OpMap = {
#    'RealDiv': Debug,
#    'Fill': Debug,
    'Pad': Pad,
    'Add': Add,
    'Sub': Sub,
    'Mul': Mul,
    'Mean': Mean,
    'Relu': ReLU,
    'AddV2': Add,
    'Relu6': ReLUX,
    'MaxPool': Pool,
    'AvgPool': Pool,
    'MatMul': MatMul,
    'BiasAdd': BiasAdd,
    'Squeeze': Reshape,
    'Reshape': Reshape,
    'Softmax': Softmax,
    'ConcatV2': Concat,
    'Placeholder': Input,
    'Conv2D': Convolution,
    'LeakyRelu': LeakyRelu,
    'FusedBatchNorm': BatchNorm,
    'SpaceToDepth': SpaceToDepth,
    'FusedBatchNormV3': BatchNorm,
    'ResizeNearestNeighbor': Resize,
    'DepthwiseConv2dNative': ConvolutionDepthwise,
}


class Model(Base):

    def __init__(self, model, graph, param):
        super().__init__(model, graph)
        self.param = param
        self.inputs = []
        self.inputs_shape = []
        self.constant = dict()
        self.indentity = dict()
        self.tf_ops = []
        self.operators = []
        self.layers = []
        self.legacys = []
        self.setInited()


    def parse(self):
        logger.debug('Parsing the TensorFlow Model...')

        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(self.graph, name='')

        # Shape Inference
        graph = shape_inference(self.model, graph, getDynamicInputShape(graph), self.param)

        for op in graph.get_operations():
            if op.type == 'Const':
                self.constant[op.outputs[0].name] = tf.get_static_value(op.outputs[0])
            elif op.type == 'Identity':
                self.indentity[op.outputs[0].name] = self.indentity.get(op.inputs[0].name, op.inputs[0].name)
            elif op.type == 'Placeholder':
                self.inputs_shape.append(op.outputs[0].shape)
                self.inputs.append(op.outputs[0].name)
                self.layers.append(make_caffe_input_layer(op.outputs[0].name, shape_map_nhwc2nchw(op.outputs[0].shape), len(self.inputs), self.param))
                self.param['inputs_shape'] = self.inputs_shape
            elif op.type in ['NoOp']: #ignore_op
                pass
            else:
                self.tf_ops.append(op)

        if len(self.tf_ops) == 0:
            sys.exit('Error: Model file is not Tensorflow Model.\n')

        print('Tensorflow GraphDef Input size: (graph version=%d)' %graph.version)
        for i, shape in enumerate(self.inputs_shape):
            print(self.inputs[i], shape)

        # Parse all operations
        for index, tf_op in enumerate(self.tf_ops):
            if tf_op.type not in OpMap:
                errorMsg = 'Error: Operator [' + tf_op.type + '] does not Support.\n'
                sys.exit(errorMsg)

            op = OpMap[tf_op.type](self, tf_op, index)
            op.parse()

            if op.status.parsed:
                self.operators.append(op)
            else:
                self.legacys.append(op)

        self.setParsed()


    def convert(self):
        logger.debug('Converting the Model...')

        for op in self.operators:
            logger.debug(op)
            layers = op.convert()
            if layers is None:
                continue
            for layer in layers:
                self.layers.append(layer)

        self.setConverted()


    def save(self, caffe_model_path):
        save_caffe_model(caffe_model_path, self.layers)
