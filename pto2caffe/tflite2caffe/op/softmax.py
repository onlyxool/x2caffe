import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')

class Softmax(Operator):

    TypeMapping = {
            tflite.BuiltinOperator.SOFTMAX: 'Softmax',
    }

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.softmax_param = dict()
        self.softmax_param['axis'] = 1
        self.attrs = self.softmax_param
        self.setInited()

    @property
    def type(self):
        return 'Softmax'

    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op.InputsLength() == 1)
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()

        self.setParsed()

    def propagatableTensors(self):
        pass

    def transform(self):
        pass

    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, softmax_param=self.softmax_param)
        self.setConverted()
        return [layer]
