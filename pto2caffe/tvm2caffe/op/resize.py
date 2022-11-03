import numpy as np

from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Resize(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'image.resize2d')
        self.setInited()


    def parse(self):
        super().__parse__()

        if self.inputs_shape[0] is not None:
            import math
            input_h = self.inputs_shape[0][self.ndim('H')]
            input_w = self.inputs_shape[0][self.ndim('W')]
            output_h = self.attrs['size'][0]
            output_w = self.attrs['size'][1]
            scale_factor_h = output_h / input_h
            scale_factor_w = output_w / input_w
            if self.attrs['rounding_method'] == 'round':
                scale_factor = round(scale_factor_h if scale_factor_h == scale_factor_w else 0)
            elif self.attrs['rounding_method'] == 'floor': 
                scale_factor = math.floor(scale_factor_h if scale_factor_h == scale_factor_w else 0)
            elif self.attrs['rounding_method'] == 'ceil':
                scale_factor = math.ceil(scale_factor_h if scale_factor_h == scale_factor_w else 0)
            else:
                scale_factor = int(scale_factor_h if scale_factor_h == scale_factor_w else 0)
        else:
            scale_factor = None

        if self.attrs.get('method', 'linear') == 'nearest_neighbor':
            if scale_factor is not None and scale_factor % 1 == 0:
                self.type = 'Deconvolution'

                self.convolution_param = dict()
                self.convolution_param['bias_term'] = False
                self.convolution_param['num_output'] = self.outputs_shape[0][1]
                self.convolution_param['kernel_h'] = scale_factor
                self.convolution_param['kernel_w'] = scale_factor
                self.convolution_param['stride_h'] = scale_factor
                self.convolution_param['stride_w'] = scale_factor
                self.convolution_param['group'] = self.inputs_shape[0][1]

                # Padding
                legacy_pad = self.model.pad.get(self.relay_inputs[0], [0, 0, 0, 0])
                if legacy_pad[0] == legacy_pad[2] and legacy_pad[1] == legacy_pad[3]:
                    self.convolution_param['pad_h'] = conv_pad[0]
                    self.convolution_param['pad_w'] = conv_pad[1]
                else:
                    self.convolution_param['pad_t'] = conv_pad[0]
                    self.convolution_param['pad_l'] = conv_pad[1]
                    self.convolution_param['pad_b'] = conv_pad[2]
                    self.convolution_param['pad_r'] = conv_pad[3]

                self.attrs = self.convolution_param

                self.weight = np.ones((self.outputs_shape[0][1], 1, scale_factor, scale_factor), dtype=int)
                self.inputs_buf[1] = self.weight
                self.inputs_shape[1] = self.inputs_buf[1].shape
            else:
                self.type = 'Upsample'
                self.upsample_param = dict()
                self.upsample_param['new_height'] = self.attrs['size'][0]
                self.upsample_param['new_width'] = self.attrs['size'][1]
                self.attrs = self.upsample_param
        elif self.attrs.get('method', 'linear') == 'linear':
            self.type = 'Interp'
            self.interp_param = dict()
            self.interp_param['align_corners'] = True if self.attrs.get('coordinate_transformation_mode', 'half_pixel') == 'align_corners' else False
            self.interp_param['height'] = self.attrs['size'][0]
            self.interp_param['width'] = self.attrs['size'][1]
            self.attrs = self.interp_param
        elif self.attrs.get('method', 'linear') == 'cubic':
            raise NotImplementedError

        self.setParsed()


    def convert(self):
        if self.type == 'Deconvolution':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, None, convolution_param=self.convolution_param)
        elif self.type == 'Upsample':
            layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, upsample_param=self.upsample_param)
        elif self.type == 'Interp':
            layer = caffe_layer(self.layer_type, self.name, self.inputs[:1], self.inputs_buf, self.outputs, interp_param=self.interp_param)

        self.setConverted()

        return [layer]
