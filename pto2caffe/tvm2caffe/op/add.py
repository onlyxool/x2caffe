import numpy as np
from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator

from util import dim_map_nhwc2nchw


class Add(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'add')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is None and self.inputs_buf[1] is None and self.inputs_shape[0] == self.inputs_shape[1]:
            self.type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 1 # Caffe Eltwise SUM
            self.attrs = self.eltwise_param
            self.setParsed()
        elif (self.inputs_buf[0] is not None or self.inputs_buf[1] is not None) or (self.inputs_shape[0] != self.inputs_shape[1]):
            self.type = 'Bias'

            if self.inputs_buf[1] is not None and np.all(self.inputs_buf[1] == 0):
                self.byPassOperator()
                return

            self.bias_param = dict()
            if 'axis' in self.attrs:
                self.bias_param['axis'] = self.attrs['axis']
            else:
                self.bias_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0

            self.bias_param['num_axes'] = list(np.array(self.inputs_shape[0]) == np.array(self.inputs_shape[1])).count(True) if len(self.inputs_shape[1]) > 0 else 0

            self.attrs = self.bias_param

            self.bias = self.inputs_buf[1]

            self.setParsed()


    def convert(self):
        if self.type == 'Eltwise':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        elif self.type == 'Bias':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.bias, bias_param=self.bias_param)

        self.setConverted()

        return [layer]
