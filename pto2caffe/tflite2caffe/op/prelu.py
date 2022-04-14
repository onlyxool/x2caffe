import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')


class PReLU(Operator):

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.setInited()


    @property
    def type(self):
        return 'PReLU'


    def parse(self):
        logger.debug("Parsing %s...", self.type)
        assert(self.op_code == tflite.BuiltinOperator.PRELU)

        self.parseInput()
        self.parseOutput()

        self.slope = self.inputs_buf[1].transpose(2, 0, 1)

        # Attributes
        self.prelu_param = dict()
        self.prelu_param['channel_shared'] = True if self.slope.shape[0] == 1 else False

        self.attrs = self.prelu_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.slope, prelu_param=self.prelu_param)

        self.setConverted()

        return [layer]
