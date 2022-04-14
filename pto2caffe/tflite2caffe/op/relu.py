import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')


class ReLU(Operator):

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.setInited()


    @property
    def type(self):
        return 'ReLU'


    def parse(self):
        logger.debug("Parsing %s...", self.type)
        assert(self.op_code in [tflite.BuiltinOperator.RELU, tflite.BuiltinOperator.LEAKY_RELU])

        if self.op is not None:
            self.parseInput()
            self.parseOutput()

        # Attributes
        if self.op_code == tflite.BuiltinOperator.LEAKY_RELU:
            op_opt = self.op.BuiltinOptions()
            opt = tflite.LeakyReluOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            self.relu_param = dict()
            self.relu_param['negative_slope'] = opt.Alpha()
            self.attrs = self.relu_param
        elif self.op_code == tflite.BuiltinOperator.RELU:
            self.relu_param = dict()
            self.relu_param['negative_slope'] = 0
            self.attrs = self.relu_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)

        self.setConverted()

        return [layer]
