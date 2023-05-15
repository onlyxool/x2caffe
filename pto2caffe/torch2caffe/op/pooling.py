import torch
from torch.nn.functional import max_pool2d, avg_pool2d

from math import ceil

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Pooling(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code in ('max_pool2d', 'avg_pool2d'))
        self.setInited()


    def parse(self):
        self.type = 'Pooling'
        super().__parse__()

        self.pooling_param = dict()
        self.pooling_param['pool'] = 0 if self.operator_code == 'max_pool2d' else 1
        self.pooling_param['kernel_h'] = self.inputs_buf[1][0]
        self.pooling_param['kernel_w'] = self.inputs_buf[1][1]

        self.pooling_param['stride_h'] = self.inputs_buf[2][0]
        self.pooling_param['stride_w'] = self.inputs_buf[2][1]

        self.pooling_param['pad_h'] = self.inputs_buf[3][0]
        self.pooling_param['pad_w'] = self.inputs_buf[3][1]
        self.pooling_param['ceil_mode'] = self.inputs_buf[5]

        self.attrs = self.pooling_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, [self.inputs[0]], self.inputs_buf, self.outputs, self.weight, self.bias, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]


    def forward(self):
        if self.operator_code == 'max_pool2d':
            output = max_pool2d(self.model.variable[self.inputs[0]], kernel_size=self.inputs_buf[1], stride=self.inputs_buf[2],
                    padding=self.inputs_buf[3], dilation=self.inputs_buf[4], ceil_mode=self.inputs_buf[5], return_indices=False)
        elif self.operator_code == 'avg_pool2d':
            output = avg_pool2d(self.model.variable[self.inputs[0]], kernel_size=self.inputs_buf[1], stride=self.inputs_buf[2],
                    padding=self.inputs_buf[3], dilation=self.inputs_buf[4], ceil_mode=self.inputs_buf[5], return_indices=False)

        self.model.variable[self.outputs[0]] = output
        self.model.tensor_shape[self.outputs[0]] = list(output.shape)
