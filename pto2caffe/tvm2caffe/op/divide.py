import numpy as np

from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Divide(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'divide')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is None and self.inputs_buf[1] is not None:
            self.type = 'Scale'

            # Weight & Bias
            if type(self.inputs_buf[1]) is np.ndarray:
                self.inputs_buf[1] = self.inputs_buf[1].squeeze()
                self.inputs_shape[1] = self.inputs_buf[1].shape

            self.weight = 1/self.inputs_buf[1]
            self.bias = None

            # Scale Parameter
            self.scale_param = dict()
            self.scale_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if np.ones(self.inputs_shape[1]).size > 1 else 0
            self.scale_param['num_axes'] = len(self.inputs_shape[1])
            self.scale_param['bias_term'] = False

            self.attrs = self.scale_param

            self.setParsed()
        else:
            self.unSupported('Can\'t Support Operand[1] == {}.'.format(self.inputs_buf[1]))


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()

        return [layer]
