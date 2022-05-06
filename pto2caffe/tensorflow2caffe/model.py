import sys
import logging
from dump import Dump
from base import Base
import tensorflow as tf

from tensorflow2caffe.op.pad import Pad
from tensorflow2caffe.op.add import Add
from tensorflow2caffe.op.sub import Sub
from tensorflow2caffe.op.mul import Scale
from tensorflow2caffe.op.pool import Pool
from tensorflow2caffe.op.relux import ReLUX
from tensorflow2caffe.op.resize import Resize
from tensorflow2caffe.op.concat import Concat
from tensorflow2caffe.op.matmul import MatMul
from tensorflow2caffe.op.reshape import Reshape
from tensorflow2caffe.op.softmax import Softmax
from tensorflow2caffe.op.placeholder import Input
from tensorflow2caffe.op.conv2d import Convolution
from tensorflow2caffe.op.batchnorm import BatchNorm
from tensorflow2caffe.op.leakyrelu import LeakyRelu
from tensorflow2caffe.op.spacetodepth import SpaceToDepth
from tensorflow2caffe.op.depthwise import ConvolutionDepthwise

from caffe_transform import save_caffe_model
from caffe_transform import make_caffe_input_layer
from util import shape_map_nhwc2nchw


logger = logging.getLogger('TensorFlow2Caffe')

OpMap = {
    'Pad': Pad,
    'Add': Add,
    'Sub': Sub,
    'Mul': Scale,
    'AddV2': Add,
    'Relu6': ReLUX,
    'MaxPool': Pool,
    'AvgPool': Pool,
    'MatMul': MatMul,
    'BiasAdd': Scale,
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

#ignore_op = ['Enter', 'Merge', 'Cast']

class Model(Base):

    def __init__(self, pb_file, graph, param):
        super().__init__(None, graph)
        self.model_file = pb_file
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

        for op in graph.get_operations():
            if op.type == 'Const':
                self.constant[op.outputs[0].name] = tf.get_static_value(op.outputs[0])
            elif op.type == 'Identity':# or op.type == 'Cast':
                self.indentity[op.outputs[0].name] = self.indentity.get(op.inputs[0].name, op.inputs[0].name)
            elif op.type == 'Placeholder':
                input_shape = []
                for dim in op.get_attr('shape').dim:
                    input_shape.append(dim.size if dim.size != -1 else 1)
                self.inputs_shape.append(input_shape)
                self.inputs.append(op.outputs[0].name)
                self.layers.append(make_caffe_input_layer(op.outputs[0].name, shape_map_nhwc2nchw(input_shape), len(self.inputs), self.param))
            else:
                self.tf_ops.append(op)
        self.param['inputs_shape'] = self.inputs_shape

        if len(self.tf_ops) == 0:
            sys.exit('Error: Model file is not Tensorflow Model.\n')

        print('Tensorflow GraphDef Input size: (graph version=%d)' %graph.version)
        for i, shape in enumerate(self.inputs_shape):
            print(self.inputs[i], shape)

        # Parse all operations
        for index, tf_op in enumerate(self.tf_ops):
#            print(tf_op.type)
#            if tf_op.type in ignore_op:
#                continue

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


    def save(self, caffe_path):
        save_caffe_model(caffe_path, self.layers)


    def dump(self, model_byte, model_name, input_tensor, dump_level=-1):
        dump = Dump('TensorFlow', model_byte, model_name, input_tensor, self.param, dump_level)
        from progress_bar import ProgressBar
        progressBar = ProgressBar(len(self.operators), 0, "TensorFlow dump processing")
        for i, op in enumerate(self.operators):
            dump.operator(op)
            progressBar.setValue(i)
        progressBar.onCancel()
