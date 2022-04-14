import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')


class Sigmoid(Operator):

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.setInited()


    @property
    def type(self):
        return 'Sigmoid'


    def parse(self):
        logger.debug("Parsing %s...", self.type)
        assert(self.op_code == tflite.BuiltinOperator.LOGISTIC)

        self.parseInput()
        self.parseOutput()

        # Attributes
        self.sigmoid_param = dict()
        self.attrs = self.sigmoid_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, sigmoid_param=self.sigmoid_param)

        self.setConverted()

        return [layer]
