import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')

class Swish(Operator):

    TypeMapping = {
            tflite.BuiltinOperator.HARD_SWISH: 'HardSwish',
    }

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)

        self.swish_param = dict()

        self.setInited()


    @property
    def type(self):
        return 'HardSwish'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op.InputsLength() == 1)
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()

        self.attrs = self.swish_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, swish_param=self.swish_param)

        self.setConverted()

        return [layer]
