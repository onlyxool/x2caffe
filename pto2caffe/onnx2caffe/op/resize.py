from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator
from onnx2caffe.utility import computePad
from util import isShapeFullyDefined


class Resize(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Resize')
        self.setInited()


    def parse(self):
        super().__parse__()

        scale = list(self.inputs_buf[1]) if max(self.model.opset) <= 10 else self.inputs_buf[2]
        sizes = list(self.inputs_buf[3]) if len(self.inputs) == 4 else None

        if not isinstance(self.inputs_shape[0], list):
            self.unSupported('Can\'t Support inputs Shape: ' + str(self.inputs_shape[0]))
            return

        if hasattr(scale, '__iter__') and len(scale) == len(self.inputs_shape[0]):
            scale_factor = int(scale[2] if scale[2] == scale[3] else 0)
            if self.outputs_shape[0] is None or self.outputs_shape[0] == []:
                self.outputs_shape[0] = [int(a * b) for a, b in zip(self.inputs_shape[0], scale)]
        else:
            if self.outputs_shape[0] == [] or self.outputs_shape is None:
                self.outputs_shape[0] = sizes

            if not isShapeFullyDefined(self.outputs_shape[0]):
                self.unSupported('Can\'t Support Output Shape: ' + str(self.outputs_shape[0]))
                return

            input_h = self.inputs_shape[0][2]
            input_w = self.inputs_shape[0][3]
            output_h = self.outputs_shape[0][2]
            output_w = self.outputs_shape[0][3]
            scale_factor_h = output_h / input_h
            scale_factor_w = output_w / input_w
            scale_factor = int(scale_factor_h if scale_factor_h == scale_factor_w else 0)

        self.mode = str(self.attrs['mode'], encoding = "utf8")
        coordinate = str(self.attrs.get('coordinate_transformation_mode', b''), encoding = "utf8")
        if self.mode == 'nearest':
            if scale_factor % 1 == 0:
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

                # Padding
                legacy_pad = self.model.pad.get(self.node.input[0], {'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
                pad_dict = dict(auto_pad='SAME_LOWER'.encode())
                padding = computePad('Deconvolution', pad_dict, self.inputs_shape[0], self.outputs_shape[0], [scale_factor, scale_factor], [scale_factor, scale_factor], legacy_pad)
                self.convolution_param.update(padding)
                self.attrs = self.convolution_param

                import numpy as np
                self.weight = np.ones((self.outputs_shape[0][1], 1, scale_factor, scale_factor), dtype=int)
                self.inputs_buf[1] = self.weight
                self.inputs_shape[1] = self.inputs_buf[1].shape
            else:
                # Upsample Layer
                self.type = 'Upsample'

                # Attributes
                self.upsample_param = dict()
                self.upsample_param['scale'] = scale_factor
                self.attrs = self.upsample_param
        elif self.mode == 'bilinear' or self.mode == 'linear':
            # Interp Layer
            self.type = 'Interp'

            # Attributes
            self.interp_param = dict()
            self.interp_param['align_corners'] = True if coordinate == 'align_corners' else False
            self.interp_param['height'] = self.outputs_shape[0][2]
            self.interp_param['width'] = self.outputs_shape[0][3]
            self.attrs = self.interp_param

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
