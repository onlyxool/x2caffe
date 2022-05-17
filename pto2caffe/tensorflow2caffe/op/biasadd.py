import numpy as np

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class BiasAdd(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'BiasAdd')
        self.setInited()


    def parse(self):
        self.layer_type = 'Scale'
        super().__parse__()

        self.weight = np.ones(self.inputs_shape[1], dtype=float, order='C')
        self.bias = self.inputs_buf[1]

        self.scale_param = dict()
        if self.inputs_shape[1] != []:
            self.scale_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0])
        self.scale_param['bias_term'] = True

        self.attrs = self.scale_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()

        return [layer]
