import logging
from dump import Dump
from base import Base
import tensorflow as tf

from tensorflow2caffe.op.pad import Pad
from tensorflow2caffe.op.add import Add
from tensorflow2caffe.op.mul import Scale
from tensorflow2caffe.op.pool import Pool
from tensorflow2caffe.op.resize import Resize
from tensorflow2caffe.op.concat import Concat
from tensorflow2caffe.op.placeholder import Input
from tensorflow2caffe.op.conv2d import Convolution
from tensorflow2caffe.op.batchnorm import BatchNorm
from tensorflow2caffe.op.leakyrelu import LeakyRelu
from tensorflow2caffe.op.spacetodepth import SpaceToDepth

from caffe_transform import save_caffe_model
from caffe_transform import make_caffe_input_layer

logger = logging.getLogger('TensorFlow2Caffe')

OpMap = {
    'Pad': Pad,
    'Add': Add,
    'Mul': Scale,
    'AddV2': Add,
    'MaxPool': Pool,
    'BiasAdd': Scale,
    'ConcatV2': Concat,
    'Placeholder': Input,
    'Conv2D': Convolution,
    'LeakyRelu': LeakyRelu,
    'FusedBatchNormV3': BatchNorm,
    'ResizeNearestNeighbor': Resize,
    'SpaceToDepth': SpaceToDepth,
}


class Model(Base):

    def __init__(self, graph, param):

        super().__init__(None, graph)
        self.graph = graph
        self.param = param
        self.inputs = []
        self.inputs_shape = []
        self.constant = dict()
        self.indentity = dict()
        self.operators = []
        self.layers = []
        self.legacys = []
        self.setInited()


    def parse(self):
        logger.debug('Parsing the TensorFlow Model...')

        tf_ops = []
        with tf.compat.v1.Session() as sess:
            # Get Ops
            sess.graph.as_default()
            tf.compat.v1.import_graph_def(self.graph, name='')
            tf_ops = [op for op in sess.graph.get_operations() if op.type != 'Const' and op.type != 'Identity']

            # Graph Input
            for op in tf_ops:
                if op.type == 'Placeholder':
                    input_shape = []
                    for dim in op.get_attr('shape').dim:
                        input_shape.append(dim.size)
                    self.inputs_shape.append(input_shape)
                    self.inputs.append(op.outputs[0].name)
                    self.layers.append(make_caffe_input_layer(op.outputs[0], input_shape, len(self.inputs), self.param))

            print('Tensorflow Frozen Graph Input size: ')
            for i, shape in enumerate(self.inputs_shape):
                print(self.inputs[i], shape)

            # Get Indentity
            indentity_ops = [op for op in sess.graph.get_operations() if op.type == 'Identity']
            for indentity_op in indentity_ops:
                self.indentity[indentity_op.outputs[0].name] = indentity_op.inputs[0].name

            # Get Const
            constant_ops = [op for op in sess.graph.get_operations() if op.type == 'Const']
            for constant_op in constant_ops:
                value = sess.run(constant_op.outputs[0])
                self.constant[constant_op.outputs[0].name] = value

        # Parse all operations
        for index, tf_op in enumerate(tf_ops):
            op = OpMap[tf_op.type](self, tf_op, tf_op.type, index)
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
            for layer in layers:
                self.layers.append(layer)

        self.setConverted()


    def save(self, caffe_name, caffe_path):
        save_caffe_model(caffe_name, caffe_path, self.layers)


    def dump(self, model_byte, model_name, input_tensor, dump_level=-1):
        dump = Dump('TensorFlow', model_byte, model_name, input_tensor, self.param, dump_level)
        from progress_bar import ProgressBar
        progressBar = ProgressBar(len(self.operators), 0, "TensorFlow dump processing")
        for i, op in enumerate(self.operators):
            dump.operator(op)
            progressBar.setValue(i)
        progressBar.onCancel()
