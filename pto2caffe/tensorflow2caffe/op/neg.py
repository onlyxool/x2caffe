import numpy as np
from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Neg(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Neg')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0] * -1)
        else:
            self.type = 'Scale'

            self.scale_param = dict()
            self.scale_param['bias_term'] = False
            self.scale_param['axis'] = 0

            self.weight = np.ones(self.inputs_shape[0]) * -1
            self.bias = None

            self.attrs = self.scale_param
            self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()

        return layer
