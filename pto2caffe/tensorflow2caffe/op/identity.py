import logging

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

logger = logging.getLogger('TensorFlow2Caffe')

class Identity(Operator):

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.input_param = dict()
        self.attrs = self.input_param
        self.setInited()

    @property
    def type(self):
        return 'Input'

    def parse(self):
        logger.debug("Parsing %s...", self.type)
        self.parseInput()
        self.parseOutput()

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, input_param=self.input_param)

        self.setConverted()

        return [layer]
