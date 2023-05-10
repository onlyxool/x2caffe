import numpy as np

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class UpsampleNearest(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'upsample_nearest2d')
        self.setInited()


    def compute_output_shape(self):
        if not self.isInputShapeFullyDefined(0):
            self.unSupported('Illegal Input Shape.')
            return

        outputs_shape = self.inputs_shape[0][:2]+self.inputs_buf[1] if self.inputs_buf[1] is not None else list((np.array(self.inputs_shape[0]) * np.array([1,1]+self.inputs_buf[2])).astype(int))
        self.model.tensor_shape[self.outputs[0]] = self.outputs_shape[0] = outputs_shape


    def parse(self):
        super().__parse__()

        self.compute_output_shape()

        output_size = self.inputs_buf[1]
        scale_factors = self.inputs_buf[2]

        scale_factor_h = int(scale_factors[0] if output_size is None else int(output_size[0] / self.inputs_shape[0][2]))
        scale_factor_w = int(scale_factors[1] if output_size is None else int(output_size[1] / self.inputs_shape[0][3]))

        scale_factor = scale_factor_h if scale_factor_h == scale_factor_w else 0
        assert(scale_factor != 0)

        if scale_factor % 1 == 0:
            self.type = 'Deconvolution'

            self.convolution_param = dict()
            self.convolution_param['bias_term'] = False
            self.convolution_param['num_output'] = self.outputs_shape[0][1]
            self.convolution_param['kernel_h'] = scale_factor_h
            self.convolution_param['kernel_w'] = scale_factor_w
            self.convolution_param['stride_h'] = int(scale_factor_h)
            self.convolution_param['stride_w'] = int(scale_factor_w)
            self.convolution_param['group'] = self.inputs_shape[0][1]

            self.weight = np.ones((self.outputs_shape[0][1], 1, int(scale_factor), int(scale_factor)), dtype=int)
            self.inputs_buf[1] = self.weight
            self.inputs_shape[1] = list(self.weight.shape)


            self.attrs = self.convolution_param
        else:
            self.type = 'Upsample'
            self.upsample_param = dict()
            self.upsample_param['scale'] = scale_factor
            self.attrs = self.upsample_param

        self.setParsed()


    def convert(self):
        if self.type == 'Deconvolution':
            layer = caffe_layer(self.layer_type, self.name, self.inputs[:2], self.inputs_buf, self.outputs, self.weight, None, convolution_param=self.convolution_param)
        elif self.type == 'Upsample':
            layer = caffe_layer(self.layer_type, self.name, self.inputs[:1], self.inputs_buf, self.outputs, upsample_param=self.upsample_param)

        self.setConverted()

        return [layer]
