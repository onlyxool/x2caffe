import tflite

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class LeakyReLU(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code == 'LEAKY_RELU')
        self.setInited()


    def parse(self):
        self.type = 'ReLU'
        super().__parse__()

        op_opt = self.op.BuiltinOptions()
        opt = tflite.LeakyReluOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)

        # Attributes
        self.relu_param = dict()
        self.relu_param['negative_slope'] = opt.Alpha()

        self.attrs = self.relu_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)

        self.setConverted()

        return [layer]
