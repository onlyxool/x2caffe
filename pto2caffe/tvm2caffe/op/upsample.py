import numpy as np

from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Upsample(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'nn.upsampling')
        self.setInited()


    def parse(self):
        super().__parse__()

        # scale_factor
        scale_factor_height = round(self.attrs['scale_h'])
        scale_factor_width = round(self.attrs['scale_w'])

        if scale_factor_height == scale_factor_width:
            scale_factor = scale_factor_width
        else:
            raise NotImplementedError

        if scale_factor % 1 == 0 and self.outputs_shape[0] is not None:
            # Deconvolution Layer
            self.type = 'Deconvolution'

            # Attributes
            self.convolution_param = dict()
            self.convolution_param['bias_term'] = False
            self.convolution_param['num_output'] = self.outputs_shape[0][1]
            self.convolution_param['kernel_h'] = scale_factor
            self.convolution_param['kernel_w'] = scale_factor
            self.convolution_param['stride_h'] = scale_factor
            self.convolution_param['stride_w'] = scale_factor
            self.convolution_param['group'] = self.inputs_shape[0][1]

            self.weight = np.ones((self.outputs_shape[0][1], 1, scale_factor, scale_factor), dtype=np.float32)
            self.inputs_buf.append(self.weight)
            self.inputs_shape.append(self.inputs_buf[1].shape)

            # Padding
            conv_pad = self.model.pad.get(self.relay_inputs[0], [0, 0, 0, 0]) 
            if conv_pad[0] == conv_pad[2] and conv_pad[1] == conv_pad[3]:
                self.convolution_param['pad_h'] = conv_pad[0]
                self.convolution_param['pad_w'] = conv_pad[1]
            else:
                self.convolution_param['pad_t'] = conv_pad[0]
                self.convolution_param['pad_l'] = conv_pad[1]
                self.convolution_param['pad_b'] = conv_pad[2]
                self.convolution_param['pad_r'] = conv_pad[3]

            self.attrs = self.convolution_param
        else:
            # Upsample Layer
            self.type = 'Upsample'

            # Attributes
            self.upsample_param = dict()
            self.upsample_param['scale'] = scale_factor
            self.attrs = self.upsample_param

        self.setParsed()


    def convert(self):
        if self.type == 'Deconvolution':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, None, convolution_param=self.convolution_param)
        elif self.type == 'Upsample':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, upsample_param=self.upsample_param)

        self.setConverted()

        return [layer]
