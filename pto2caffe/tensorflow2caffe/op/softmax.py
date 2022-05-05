import logging

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

logger = logging.getLogger('TensorFlow2Caffe')


class Softmax(Operator):

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.softmax_param = dict()
        self.setInited()


    @property
    def type(self):
        return 'Softmax'


    def parse(self):
        logger.debug('Parsing %s...', self.type)

        self.parseInput()
        self.parseOutput()
        self.parseAttributes()

        self.attrs = self.softmax_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, softmax_param=self.softmax_param)

        self.setConverted()

        return [layer]
