import logging
from dump import Dump
from base import Base
import tensorflow as tf

from tensorflow2caffe.op.pool import Pool
from tensorflow2caffe.op.binary import Binary
from tensorflow2caffe.op.placeholder import Input
from tensorflow2caffe.op.conv2d import Convolution
from tensorflow2caffe.op.batchnorm import BatchNorm
from tensorflow2caffe.op.leakyrelu import LeakyRelu

#from tflite2caffe.op.pad import Pad
#from tflite2caffe.op.swish import Swish
#from tflite2caffe.op.split import Slice #TODO
#from tflite2caffe.op.binary import Binary
#from tflite2caffe.op.reduce import Reduce
#from tflite2caffe.op.resize import Resize
#from tflite2caffe.op.concat import Concat
#from tflite2caffe.op.reshape import Reshape
#from tflite2caffe.op.pooling import Pooling
#from tflite2caffe.op.softmax import Softmax

#from tflite2caffe.op.deconv import Deconvolution
#from tflite2caffe.op.quantize import Quantize
#from tflite2caffe.op.transpose import Permute
#from tflite2caffe.op.activation import Activation
#from tflite2caffe.op.fullyconnected import InnerProduct
#from tflite2caffe.op.activation import handleFusedActivation

from caffe_transform import save_caffe_model


logger = logging.getLogger('TensorFlow2Caffe')

OpMap = {
    'Add': Binary,
    'MaxPool': Pool,
    'Placeholder': Input,
    'Conv2D': Convolution,
    'LeakyRelu': LeakyRelu,
    'FusedBatchNormV3': BatchNorm,
#    'Const': Const,
#    'PAD': Pad,
#    'MUL': Binary,
#    'SUB': Binary,
#    'SPLIT': Slice,
#    'MEAN': Reduce,
#    'RESHAPE': Reshape,
#    'SQUEEZE': Reshape,
#    'SOFTMAX': Softmax,
#    'RELU': Activation,
#    'PRELU': Activation,
#    'HARD_SWISH': Swish,
#    'QUANTIZE': Quantize,
#    'TRANSPOSE': Permute,
#    'CONV_2D': Convolution,
#    'LOGISTIC': Activation,
#    'DEQUANTIZE': Quantize,
#    'MAX_POOL_2D': Pooling,
#    'STRIDED_SLICE': Slice,
#    'CONCATENATION': Concat,
#    'REDUCE_MAX': Reduce,
#    'LEAKY_RELU': Activation,
#    'RESIZE_BILINEAR': Resize,
#    'AVERAGE_POOL_2D': Pooling,
#    'TRANSPOSE_CONV': Deconvolution,
#    'FULLY_CONNECTED': InnerProduct,
#    'DEPTHWISE_CONV_2D': Convolution,
#    'RESIZE_NEAREST_NEIGHBOR': Resize,
##    'DIV': Binary,
##    'MIRROR_PAD': Pad,
}


class Model(Base):

    def __init__(self, pb_file, param):

        super().__init__(None, None)
        self.pb_file = pb_file 
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
            with tf.io.gfile.GFile(self.pb_file, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())

            # Get Ops
            sess.graph.as_default()
            tf.compat.v1.import_graph_def(graph_def, name='')
            tf_ops = [op for op in sess.graph.get_operations() if op.type != 'Const' and op.type != 'Identity']

            # Graph Input
            for op in tf_ops:
                if op.type == 'Placeholder':
                    input_shape = []
                    for dim in op.get_attr('shape').dim:
                        input_shape.append(dim.size)
                    self.inputs_shape.append(input_shape)
                    self.inputs.append(op.outputs[0].name)

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
