import numpy as np

from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class ConvTranspose(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'nn.conv2d_transpose')
        self.setInited()


    def parse(self):
        self.type = 'Deconvolution'
        super().__parse__()

        if self.attrs.get('kernel_layout', 'IOHW') == 'IOHW':
            self.weight = self.inputs_buf[1]
        else:
            kernel_layout = self.attrs['kernel_layout'] if 'kernel_layout' in self.attrs else 'None'
            self.unSupported('Can\'t support kernel_layout == ' + kernel_layout)
            return

        self.inputs_buf[1] = self.weight

        # Bias
        self.bias = None

        # Attribute
        self.convolution_param = dict()
        self.convolution_param['num_output'] = self.attrs['channels']
        self.convolution_param['stride_h'] = self.attrs.get('strides', [1, 1])[0]
        self.convolution_param['stride_w'] = self.attrs.get('strides', [1, 1])[1]
        self.convolution_param['dilation'] = self.attrs.get('dilation', [1, 1])
        self.convolution_param['kernel_h'] = self.attrs['kernel_size'][0]
        self.convolution_param['kernel_w'] = self.attrs['kernel_size'][1]
        self.convolution_param['bias_term'] = True if self.bias is not None else False

        # Padding
        legacy_pad = self.model.pad.get(self.relay_inputs[0], [0, 0, 0, 0]) 
        attr_pad = self.attrs.get('padding', [0, 0, 0, 0]) # t, l, b, r
        conv_pad = (np.array(legacy_pad) + np.array(attr_pad)).tolist()
        if conv_pad[0] == conv_pad[2] and conv_pad[1] == conv_pad[3]:
            self.convolution_param['pad_h'] = conv_pad[0]
            self.convolution_param['pad_w'] = conv_pad[1]
        else:
            self.convolution_param['pad_t'] = conv_pad[0]
            self.convolution_param['pad_l'] = conv_pad[1]
            self.convolution_param['pad_b'] = conv_pad[2]
            self.convolution_param['pad_r'] = conv_pad[3]

        self.attrs = self.convolution_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, convolution_param=self.convolution_param)

        self.setConverted()

        return [layer]
