import logging
import numpy as np

from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator

logger = logging.getLogger('TensorFlow2Caffe')

class Resize(Operator):

    def __init__(self, model, tf_op, tf_op_code, index):
        super().__init__(model, tf_op, tf_op_code, index)

        self.setInited()


    @property
    def type(self):
        if self.op_code == 'ResizeNearestNeighbor':
            if hasattr(self, 'convolution_param'):
                return 'Deconvolution'
            else:
                return 'Upsample'
        elif self.op_code == 'ResizeBilinear':
            return 'Interp'
        else:
            raise NotImplementedError


    def parse(self):
        logger.debug('Parsing %s...', self.type)
        self.parseInput()
        self.parseOutput()
        self.parseAttributes()

        # Output shape
        output_h = self.inputs_buf[1][0]
        output_w = self.inputs_buf[1][1]

        # Input Shape
        input_h = self.inputs_shape[0][2]
        input_w = self.inputs_shape[0][3]

        scale_factor = output_h/input_h
        if self.op_code == 'ResizeNearestNeighbor':
            if scale_factor % 1 == 0:
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

                self.attrs = self.convolution_param
            else:
                self.upsample_param = dict()
                self.upsample_param['scale'] = scale_factor
                self.attrs = self.upsample_param
        elif self.op_code == 'ResizeBilinear':
            self.interp_param = dict()
            self.interp_param['align_corners'] = self.attrs['align_corners']
            self.interp_param['height'] = self.inputs_buf[1][0]
            self.interp_param['width'] = self.inputs_buf[1][1]
        else:
            raise NotImplementedError


        self.setParsed()


    def convert(self):
        if self.op_code == 'ResizeNearestNeighbor':
            if hasattr(self, 'convolution_param'):
                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, None, convolution_param=self.convolution_param)
            elif hasattr(self, 'upsample_param'):
                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, upsample_param=self.upsample_param)
        elif self.op_code == 'ResizeBilinear':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, interp_param=self.interp_param)

        self.setConverted()

        return [layer]
