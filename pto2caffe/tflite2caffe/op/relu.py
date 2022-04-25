import tflite

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class ReLU(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator in ('RELU', 'LEAKY_RELU'))
        self.setInited()


    def parse(self):
        self.layer_type = 'ReLU'

        if self.op is not None:
            self.parseInputOutput()

        # Attributes
        if self.operator == 'LEAKY_RELU':
            op_opt = self.op.BuiltinOptions()
            opt = tflite.LeakyReluOptions()
            opt.Init(op_opt.Bytes, op_opt.Pos)
            self.relu_param = dict()
            self.relu_param['negative_slope'] = opt.Alpha()
            self.attrs = self.relu_param
        elif self.operator == 'RELU':
            self.relu_param = dict()
            self.relu_param['negative_slope'] = 0
            self.attrs = self.relu_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)

        self.setConverted()

        return [layer]
