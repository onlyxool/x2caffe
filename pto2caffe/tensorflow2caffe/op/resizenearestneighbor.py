import numpy as np

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

from util import handleLegacyPad


class ResizeNearestNeighbor(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'ResizeNearestNeighbor')
        self.setInited()


    def parse(self):
        super().__parse__()

        # Output shape
        if self.inputs_buf[1] is not None:
            output_h = self.inputs_buf[1][0]
            output_w = self.inputs_buf[1][1]
        elif self.outputs_shape[0] is not None:
            output_h = self.outputs_shape[0][-2]
            output_w = self.outputs_shape[0][-1]
        else:
            raise ValueError(self.op.name)

        # Input Shape
        input_h = self.inputs_shape[0][2]
        input_w = self.inputs_shape[0][3]

        scale_factor = output_h/input_h

        if scale_factor % 1 == 0:
            self.layer_type = 'Deconvolution'
            self.convolution_param = dict()
            self.convolution_param['bias_term'] = False
            self.convolution_param['num_output'] = self.outputs_shape[0][1]
            self.convolution_param['kernel_h'] = int(scale_factor)
            self.convolution_param['kernel_w'] = int(scale_factor)
            self.convolution_param['stride_h'] = int(scale_factor)
            self.convolution_param['stride_w'] = int(scale_factor)
            self.convolution_param['group'] = self.inputs_shape[0][1]

            self.weight = np.ones((self.outputs_shape[0][1], 1, int(scale_factor), int(scale_factor)), dtype=int)
            self.inputs_buf[1] = self.weight
            self.inputs_shape[1] = self.weight.shape

            # Padding
            legacy_pad = self.model.pad.get(self.op.inputs[0].name, {'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
            padding = handleLegacyPad('VALID', self.inputs_shape[0], self.outputs_shape[0], self.convolution_param, legacy_pad, self.type)
            self.convolution_param.update(padding)

            self.attrs = self.convolution_param
        else:
            self.layer_type = 'Upsample'
            self.upsample_param = dict()
            self.upsample_param['scale'] = scale_factor
            self.attrs = self.upsample_param

        self.setParsed()


    def convert(self):
        if self.type == 'Deconvolution':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, None, convolution_param=self.convolution_param)
        elif self.type == 'Upsample':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, upsample_param=self.upsample_param)

        self.setConverted()

        return [layer]
