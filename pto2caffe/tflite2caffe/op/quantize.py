import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')


class Quantize(Operator):

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)

        self.setInited()


    @property
    def type(self):
        if hasattr(self, 'reshape_param'):
            return 'Reshape'
        elif self.op_code == tflite.BuiltinOperator.QUANTIZE:
            return 'Quantize'
        elif self.op_code == tflite.BuiltinOperator.DEQUANTIZE:
            return 'Dequantize'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        assert(self.op_code in (tflite.BuiltinOperator.QUANTIZE, tflite.BuiltinOperator.DEQUANTIZE))

        self.parseInput()
        self.parseOutput()

        if self.inputs_buf[0] is None:
            self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
            self.setParsed()
        else:
            self.model.tensor[self.outputs[0]] = self.model.tensor[self.inputs[0]]


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)

        self.setConverted()

        return [layer]
