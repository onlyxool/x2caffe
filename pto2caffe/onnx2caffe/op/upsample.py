# This operator is deprecated in opset version 10
# Ref: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Upsample

import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator
from onnx2caffe.utility import computePad


class Upsample(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Upsample')
        self.setInited()


    def parse(self):
        super().__parse__()

        # scale_factor
        if self.model.opset[0] < 7:
            scale_factor_height = int(self.attrs.get('height_scale'))
            scale_factor_width = int(self.attrs.get('width_scale'))
        elif self.model.opset[0] >= 7 and self.model.opset[0] < 9:
            scale_factor_height = int(self.attrs['scales'][2])
            scale_factor_width = int(self.attrs['scales'][3])
        elif self.model.opset[0] >= 9:
            scale_factor_height = int(self.inputs_buf[1][2])
            scale_factor_width = int(self.inputs_buf[1][3])

        if scale_factor_height == scale_factor_width:
            scale_factor = scale_factor_width
        else:
            raise NotImplementedError

        if scale_factor % 1 == 0:
            # Deconvolution Layer
            self.layer_type = 'Deconvolution'

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
            if self.model.opset[0] < 9:
                self.inputs_buf.append(self.weight)
                self.inputs_shape.append(self.inputs_buf[1].shape)
            else:
                self.inputs_buf[1] = self.weight
                self.inputs_shape[1] = self.inputs_buf[1].shape

            # Padding
            legacy_pad = self.model.pad.get(self.node.input[0], {'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
            padding = computePad(self.type, self.attrs, self.inputs_shape[0], self.outputs_shape[0], [scale_factor, scale_factor], [scale_factor, scale_factor], legacy_pad)
            self.convolution_param.update(padding)

            self.attrs = self.convolution_param
        else:
            # Upsample Layer
            self.layer_type = 'Upsample'

            # Attributes
            self.upsample_param = dict()
            self.upsample_param['scale'] = scale_factor
            self.attrs = self.upsample_param

        self.setParsed()


    def convert(self):
        if self.type == 'Deconvolution':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, None, convolution_param=self.convolution_param)
        elif self.type == 'Upsample':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, upsample_param=self.upsample_param)
        else:
            raise NotImplementedError

        self.setConverted()

        return [layer]
