import numpy as np

from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class Upsample(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code in ('nn.Upsample', 'F.upsample'))
        self.setInited()


    def parse(self):
        super().__parse__()

        mode = self.attrs['mode']
        scale_factor_h = int(self.outputs_shape[0][2] / self.inputs_shape[0][2])
        scale_factor_w = int(self.outputs_shape[0][3] / self.inputs_shape[0][3])
        scale_factor = scale_factor_h if scale_factor_h == scale_factor_w else 0
        assert(scale_factor != 0)

        if mode == 'nearest':
            if scale_factor % 1 == 0:
                # Deconvolution Layer
                self.layer_type = 'Deconvolution'

                # Attributes
                self.convolution_param = dict()
                self.convolution_param['bias_term'] = False
                self.convolution_param['num_output'] = self.outputs_shape[0][1]
                self.convolution_param['kernel_size'] = int(scale_factor)
                self.convolution_param['stride_h'] = int(scale_factor)
                self.convolution_param['stride_w'] = int(scale_factor)
                self.convolution_param['group'] = self.inputs_shape[0][1]
                self.attrs = self.convolution_param

                self.weight = np.ones((self.outputs_shape[0][1], 1, int(scale_factor), int(scale_factor)), dtype=int)
                self.inputs_buf.append(self.weight)
                self.inputs_shape.append(self.inputs_buf[1].shape)
            else:
                # Upsample Layer
                self.layer_type = 'Upsample'

                # Attributes
                self.upsample_param = dict()
                self.upsample_param['scale'] = scale_factor
                self.attrs = self.upsample_param
        elif mode == 'bilinear':
            # Interp Layer
            self.layer_type = 'Interp'
            #self.attrs['size']

            # Attributes
            self.interp_param = dict()
            self.interp_param['align_corners'] = self.attrs['align_corners']
            self.interp_param['height'] = self.outputs_shape[0][2]
            self.interp_param['width'] = self.outputs_shape[0][3]
            self.attrs = self.interp_param
        else:
            raise NotImplementedError

        self.setParsed()


    def convert(self):
        if self.type == 'Deconvolution':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, None, convolution_param=self.convolution_param)
        elif self.type == 'Upsample':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, upsample_param=self.upsample_param)
        elif self.type == 'Interp':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, interp_param=self.interp_param)

        self.setConverted()

        return [layer]
