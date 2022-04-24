import tflite

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class Reshape(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        self.reshape_param = dict()
        self.setInited()


    def parse(self):
        self.layer_type = 'Reshape'

        assert(self.operator in ('RESHAPE', 'SQUEEZE'))

        self.parseInput()
        self.parseOutput()

        # Attributes
        op_opt = self.op.BuiltinOptions()
        if self.operator == 'RESHAPE':
            opt = tflite.ReshapeOptions()
            self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
        elif self.operator == 'SQUEEZE':
            opt = tflite.SqueezeOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))

        self.attrs = self.reshape_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)

        self.setConverted()

        return [layer]
