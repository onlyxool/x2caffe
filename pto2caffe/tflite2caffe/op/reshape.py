import tflite
import logging

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator

logger = logging.getLogger('tflite2caffe')

class Reshape(Operator):

    TypeMapping = { 
        tflite.BuiltinOperator.RESHAPE: 'Reshape',
        tflite.BuiltinOperator.SQUEEZE: 'Squeeze',
    }   


    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)
        self.reshape_param = dict()
        self.setInited()


    @property
    def type(self):
        return 'Reshape'


    def parse(self):
        logger.debug("Parsing %s...", self.shorty)

        if self.op_code == tflite.BuiltinOperator.RESHAPE:
            assert(self.op.InputsLength() >= 2)
        elif self.op_code == tflite.BuiltinOperator.SQUEEZE:
            assert(self.op.InputsLength() == 1)
        assert(self.op.OutputsLength() == 1)

        self.parseInput()
        self.parseOutput()

        # Option
        op_opt = self.op.BuiltinOptions()
        if self.op_code == tflite.BuiltinOperator.RESHAPE:
            opt = tflite.ReshapeOptions()
            self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
        elif self.op_code == tflite.BuiltinOperator.SQUEEZE:
            opt = tflite.SqueezeOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
        else:
            raise NotImplementedError

        self.attrs = self.reshape_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)

        self.setConverted()

        return [layer]
