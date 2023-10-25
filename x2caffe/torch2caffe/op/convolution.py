import torch
import numpy as np
from torch.nn.functional import conv2d, conv_transpose2d

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Convolution(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'convolution')
        self.setInited()


    def parse(self):
        super().__parse__()

        self.weight = self.inputs_buf[1]
        self.bias = self.inputs_buf[2] = self.inputs_buf[2] if self.inputs_buf[2] is not None else np.zeros(self.inputs_shape[1][0])

        stride = self.inputs_buf[3]
        padding = self.inputs_buf[4]
        dilation = self.inputs_buf[5]
        transposed = self.inputs_buf[6]
        output_padding = self.inputs_buf[7]
        group = self.inputs_buf[8]

        self.type = 'Deconvolution' if transposed else 'Convolution'

        self.convolution_param = dict()
        self.convolution_param['group'] = group
        self.convolution_param['dilation'] = dilation
        self.convolution_param['stride_h'] = stride[0]
        self.convolution_param['stride_w'] = stride[1]
        self.convolution_param['kernel_h'] = self.inputs_shape[1][2]
        self.convolution_param['kernel_w'] = self.inputs_shape[1][3]
        self.convolution_param['num_output'] = self.inputs_shape[1][1] if transposed else self.inputs_shape[1][0]
        self.convolution_param['bias_term'] = True

        if output_padding == [0, 0]:
            self.convolution_param['pad_h'] = padding[0]
            self.convolution_param['pad_w'] = padding[1]
        else:
            self.convolution_param['pad_t'] = padding[0]
            self.convolution_param['pad_b'] = padding[0] - output_padding[0]
            self.convolution_param['pad_l'] = padding[1]
            self.convolution_param['pad_r'] = padding[1] - output_padding[1]

        self.attrs = self.convolution_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, convolution_param=self.convolution_param)

        self.setConverted()

        return [layer]


    def forward(self):
        if self.inputs_buf[6]:
            return conv_transpose2d(self.model.variable[self.inputs[0]], weight=torch.Tensor(self.weight), bias=torch.Tensor(self.bias),
                    stride=self.inputs_buf[3], padding=self.inputs_buf[4], output_padding=self.inputs_buf[7], groups=self.inputs_buf[8], dilation=self.inputs_buf[5])
        else:
            return conv2d(self.model.variable[self.inputs[0]], weight=torch.Tensor(self.weight), bias=torch.Tensor(self.bias),
                    stride=self.inputs_buf[3], padding=self.inputs_buf[4], dilation=self.inputs_buf[5], groups=self.inputs_buf[8])
