import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')

class Permute(Operator):

    TypeMapping = { 
            tflite.BuiltinOperator.TRANSPOSE: 'Permute',
    }

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.permute_param = dict()

        self.setInited()


    @property
    def type(self):
        return 'Permute'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op.InputsLength() == 2)
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()

        self.permute_param['order'] = list(self.inputs_buf[1])
        self.attrs = self.permute_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, permute_param=self.permute_param)

        self.setConverted()

        return [layer]
