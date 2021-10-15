import logging

from caffe_transform import caffe_layer
from caffe_transform import make_caffe_input_layer
from tensorflow2caffe.op.operator import Operator

logger = logging.getLogger('TensorFlow2Caffe')

class Input(Operator):

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.setInited()

    @property
    def type(self):
        return 'Input'

    def parse(self):
        logger.debug('Parsing %s...', self.type)
        self.parseInput()
        self.parseOutput()
        self.parseAttributes()

        self.layer = make_caffe_input_layer(self.outputs[0], self.model.param)

        self.setParsed()


    def convert(self):
        self.setConverted()

        return [self.layer]
