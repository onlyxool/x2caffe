import torch

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Matmul(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'matmul')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_buf[0] is None and self.inputs_buf[1] is not None and (len(self.inputs_shape[0]) == 2 and len(self.inputs_shape[1]) == 2):
            raise NotImplementedError
        else:
            self.type = 'MatMul'

            self.matmul_param = dict()
            self.matmul_param['transpose_a'] = 0 
            self.matmul_param['transpose_b'] = 0 
            self.attrs = self.matmul_param

            self.setParsed()


    def convert(self):
        if self.type == 'InnerProduct':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, inner_product_param=self.inner_product_param)
        elif self.type == 'MatMul':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, matmul_param=self.matmul_param)

        self.setConverted()

        return [layer]


    def forward(self):
        return torch.matmul(self.model.variable[self.inputs[0]], self.model.variable[self.inputs[1]])
