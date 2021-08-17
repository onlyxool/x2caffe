import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')

class Slice(Operator):

    TypeMapping = {
            tflite.BuiltinOperator.SPLIT: 'Slice',
    }

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.slice_param = dict()

        self.setInited()


    @property
    def type(self):
        return 'Slice'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op.InputsLength() == 2)
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()
#        print('Slice', self.inputs_shape)
#        print('Slice', self.outputs_shape)

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, slice_param=self.slice_param)

        self.setConverted()

        return [layer]
