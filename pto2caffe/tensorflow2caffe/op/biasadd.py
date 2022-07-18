import numpy as np

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class BiasAdd(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'BiasAdd')
        self.setInited()


    def parse(self):
        self.layer_type = 'Bias'
        super().__parse__()

        self.bias = self.inputs_buf[1]

        self.bias_param = dict()
        if self.inputs_shape[1] != []:
            self.bias_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0])

        if self.bias.shape != () and self.bias.shape != []:
            self.bias_param['num_axes'] = len(self.bias.shape)
        else:
            self.bias_param['num_axes'] = 0

        self.attrs = self.bias_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.bias, bias_param=self.bias_param)

        self.setConverted()

        return [layer]
