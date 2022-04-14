import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')


class ReLUX(Operator):

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.setInited()


    @property
    def type(self):
        return 'ReLUX'


    def parse(self):
        logger.debug("Parsing %s...", self.type)
        assert(self.op_code == tflite.BuiltinOperator.RELU6)

        if self.op is not None:
            self.parseInput()
            self.parseOutput()

        # Attributes
        self.relux_param = dict()
        self.relux_param['negative_slope'] = 0
        self.relux_param['x'] = 6

        self.attrs = self.relux_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relux_param=self.relux_param)

        self.setConverted()

        return [layer]
