import torch
from torch.nn.functional import adaptive_avg_pool2d

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class AdaptiveAvgPooling(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'adaptive_avg_pool2d')
        self.setInited()


    def parse(self):
        self.type = 'Pooling'
        super().__parse__()

        input_h = self.inputs_shape[0][2]
        input_w = self.inputs_shape[0][3]

        output_h = self.inputs_buf[1][0]
        output_w = self.inputs_buf[1][1]

        self.pooling_param = dict()
        self.pooling_param['pool'] = 1 

        self.pooling_param['stride_h'] = stride_h = input_h // output_h
        self.pooling_param['stride_w'] = stride_w = input_w // output_w

        self.pooling_param['kernel_h'] = input_h - (output_h - 1) * stride_h
        self.pooling_param['kernel_w'] = input_w - (output_w - 1) * stride_w

        self.attrs = self.pooling_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]


    def forward(self):
        return adaptive_avg_pool2d(self.model.variable[self.inputs[0]], self.inputs_buf[1])
