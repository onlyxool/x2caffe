# This operator is deprecated in latest version of ONNX
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
        return 'Deconvolution'


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        # Option
        self.parseAttributes()
        self.mode = str(self.attrs['mode'], encoding = "utf8")
        scale_factor = int(self.attrs.get('height_scale', self.inputs_buf[1][2]))



        self.name = 'Deconvolution' + str(self.index)
        self.convolution_param = dict()
        self.convolution_param['stride'] = scale_factor
        self.convolution_param['group'] = self.inputs_shape[0][1]
        self.convolution_param['num_output'] = self.outputs_shape[0][1]
        self.convolution_param['bias_term'] = False

        if self.mode == 'nearest':
            self.convolution_param['kernel_size'] = scale_factor
        elif self.mode == 'bilinear':
            self.convolution_param['kernel_size'] = 2 * scale_factor - scale_factor % 2
            self.convolution_param['weight_filler'] = dict(type="bilinear")
        else:
            raise NotImplementedError

        self.weight = np.ones((self.outputs_shape[0][1], 1, int(self.convolution_param['kernel_size']), int(self.convolution_param['kernel_size'])), dtype=int)
        self.inputs_buf[1] = self.weight
        self.inputs_shape[1] = self.inputs_buf[1].shape
        # TODO: self.convolution_param['pads']

        self.attrs = self.convolution_param
        self.setParsed()

    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, None, convolution_param=self.convolution_param)
        self.setConverted()
        return [layer]
