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

            if self.inputs_buf[0] is not None:
                self.inputs.reverse()
                self.inputs_shape.reverse()
                self.inputs_buf.reverse()

            if self.inputs_buf[1] is not None and np.all(self.inputs_buf[1] == 0):
                self.byPassOperator()
                return

            if type(self.inputs_buf[1]) is np.ndarray:
                self.inputs_buf[1] = self.inputs_buf[1].squeeze()
                self.inputs_shape[1] = self.inputs_buf[1].shape

            self.bias = self.inputs_buf[1]

            self.bias_param = dict()
            if 'axis' in self.attrs:
                self.bias_param['axis'] = self.attrs['axis']
                self.bias_param['axis'] = dim_map_nhwc2nchw[self.bias_param['axis']] if self.layout == 'NHWC' and len(self.inputs_shape[0]) == 4 else self.bias_param['axis']
            else:
                self.bias_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0
            self.bias_param['num_axes'] = len(self.inputs_shape[1])

            self.attrs = self.bias_param

            self.setParsed()


    def convert(self):
        if self.type == 'Eltwise':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        elif self.type == 'Bias':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.bias, bias_param=self.bias_param)

        self.setConverted()

        return [layer]
