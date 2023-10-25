import numpy as np

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Sub(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'sub')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is not None:
            self.saveConstant(self.outputs[0], self.inputs_buf[0] - self.inputs_buf[1] * self.inputs_buf[2])
        elif self.inputs_buf[0] is None and self.inputs_buf[1] is None and self.inputs_shape[0] == self.inputs_shape[1]:
            self.type = 'Eltwise'
            self.eltwise_param = dict()
            self.eltwise_param['operation'] = 3 # Caffe Eltwise SUB
            self.attrs = self.eltwise_param
            self.setParsed()
        elif self.inputs_shape[1] is not None:
            self.type = 'Bias'

            self.bias = -1 * self.inputs_buf[1] * self.inputs_buf[2]

            self.bias_param = dict()
            self.bias_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0
            self.bias_param['num_axes'] = len(self.bias.shape) if isinstance(self.bias, np.ndarray) else 0

            self.attrs = self.bias_param

            self.setParsed()


    def convert(self):
        if self.type == 'Eltwise':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, eltwise_param=self.eltwise_param)
        elif self.type == 'Bias':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.bias, bias_param=self.bias_param)

        self.setConverted()

        return [layer]
