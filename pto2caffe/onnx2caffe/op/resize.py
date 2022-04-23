import logging
import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')


class Resize(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    def parse(self):
        logger.debug("Parsing %s...", self.type)

        self.parseInput()
        self.parseOutput()

        scale = self.inputBuf_byName('scales')
        if scale is None:
            if max(self.model.opset) <= 10:
                scale = self.inputs_buf[1]
            else:
                scale = self.inputs_buf[2]

        if len(scale) >= 4:
            scale_factor = scale[2] if scale[2] == scale[3] else 0
        else:
            input_h = self.inputs_shape[0][2]
            input_w = self.inputs_shape[0][3]
            output_h = self.outputs_shape[0][2]
            output_w = self.outputs_shape[0][3]
            scale_factor_h = output_h / input_h
            scale_factor_w = output_w / input_w
            scale_factor = scale_factor_h if scale_factor_h == scale_factor_w else 0

        assert(scale_factor != 0)

        # Attributes
        self.parseAttributes()
        self.mode = str(self.attrs['mode'], encoding = "utf8")
        coordinate = str(self.attrs.get('coordinate_transformation_mode', b''), encoding = "utf8")
        if self.mode == 'nearest':
            if scale_factor % 1 == 0:
                self.layer_type = 'Deconvolution'
                self.convolution_param = dict()
                self.convolution_param['bias_term'] = False
                self.convolution_param['num_output'] = self.outputs_shape[0][1]
                self.convolution_param['kernel_size'] = int(scale_factor)
                self.convolution_param['stride_h'] = int(scale_factor)
                self.convolution_param['stride_w'] = int(scale_factor)
                self.convolution_param['group'] = self.inputs_shape[0][1]
                self.attrs = self.convolution_param
                # TODO: self.convolution_param['pads']
                self.weight = np.ones((self.outputs_shape[0][1], 1, int(scale_factor), int(scale_factor)), dtype=int)
                self.inputs_buf[1] = self.weight
                self.inputs_shape[1] = self.inputs_buf[1].shape
            else:
                self.layer_type = 'Upsample'
                self.upsample_param = dict()
                self.upsample_param['scale'] = scale_factor
                self.attrs = self.upsample_param
        elif self.mode == 'linear':
            self.layer_type = 'Interp'
            self.interp_param = dict()
            self.interp_param['align_corners'] = True if coordinate == 'align_corners' else False
            self.interp_param['height'] = self.outputs_shape[0][2]
            self.interp_param['width'] = self.outputs_shape[0][3]
            self.attrs = self.interp_param

        self.setParsed()


    def convert(self):
        if self.mode == 'nearest':
            if self.type == 'Deconvolution':
                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, None, convolution_param=self.convolution_param)
            elif self.type == 'Upsample':
                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, upsample_param=self.upsample_param)
            else:
                raise NotImplementedError
        elif self.mode == 'linear':
            layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, interp_param=self.interp_param)
        else:
            raise NotImplementedError

        self.setConverted()

        return [layer]
