# This operator is deprecated in opset version 10
# Ref: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Upsample

import logging
import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')

class Upsample(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    @property
    def type(self):
#        return 'Deconvolution'
        if hasattr(self, 'convolution_param'):
            return 'Deconvolution'
        elif hasattr(self, 'upsample_param'):
            return 'Upsample'
        elif hasattr(self, 'interp_param'):
            return 'Interp'
        else:
            return 'Resize'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Option
        self.parseAttributes()
        self.mode = str(self.attrs['mode'], encoding = "utf8")

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

        if self.mode == 'nearest':
            if scale_factor % 1 == 0:
                self.convolution_param = dict()
                self.convolution_param['stride_h'] = scale_factor
                self.convolution_param['stride_w'] = scale_factor
                self.convolution_param['group'] = self.inputs_shape[0][1]
                self.convolution_param['num_output'] = self.outputs_shape[0][1]
                self.convolution_param['bias_term'] = False
                self.convolution_param['kernel_size'] = scale_factor

                self.weight = np.ones((self.outputs_shape[0][1], 1, int(self.convolution_param['kernel_size']), int(self.convolution_param['kernel_size'])), dtype=int)
                if self.model.opset[0] < 9:
                    self.inputs_buf.append(self.weight)
                    self.inputs_shape.append(self.inputs_buf[1].shape)
                else:
                    self.inputs_buf[1] = self.weight
                    self.inputs_shape[1] = self.inputs_buf[1].shape

                self.attrs = self.convolution_param
            else:
                self.upsample_param = dict()
                self.upsample_param['scale'] = scale_factor
                self.attrs = self.upsample_param
        elif self.mode == 'bilinear' or self.mode == 'linear':
#            self.convolution_param['kernel_size'] = scale_factor
#            self.convolution_param['weight_filler'] = dict(type="bilinear")
            self.interp_param = dict()
            self.interp_param['align_corners'] = False
            self.interp_param['height'] = self.outputs_shape[0][2]
            self.interp_param['width'] = self.outputs_shape[0][3]
            self.attrs = self.interp_param
        else:
            raise NotImplementedError


        # TODO: self.convolution_param['pads']

        self.setParsed()


    def convert(self):
#        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, None, convolution_param=self.convolution_param)
        if self.mode == 'nearest':
            if hasattr(self, 'convolution_param'):
                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, None, convolution_param=self.convolution_param)
            elif hasattr(self, 'upsample_param'):
                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, upsample_param=self.upsample_param)
            else:
                raise NotImplementedError
        elif self.mode == 'bilinear' or self.mode == 'linear':
                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, interp_param=self.interp_param)
        else:
            raise NotImplementedError

        self.setConverted()

        return [layer]
