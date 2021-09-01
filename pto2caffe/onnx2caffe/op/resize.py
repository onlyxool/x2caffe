import logging
import numpy as np

from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

logger = logging.getLogger('onnx2caffe')

class Resize(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    @property
    def type(self):
        if self.op_code == 'Resize':
            if hasattr(self, 'convolution_param'):
                return 'Deconvolution'
            elif hasattr(self, 'upsample_param'):
                return 'Upsample'
            else:
                return 'nearest'
        elif self.op_code == 'Upsample':
            return 'Interp'
        else:
            raise NotImplementedError


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

        if scale[2] == scale[3]:
            scale_factor = scale[2]
        else:
            raise NotImplementedError

        # Option
        self.parseAttributes()
        self.mode = str(self.attrs['mode'], encoding = "utf8")
        coordinate = str(self.attrs.get('coordinate_transformation_mode', b''), encoding = "utf8")
        if self.mode == 'nearest':
            if scale_factor % 1 == 0:
                self.convolution_param = dict()
                self.convolution_param['bias_term'] = False
                self.convolution_param['num_output'] = self.outputs_shape[0][1]
                self.convolution_param['kernel_size'] = int(scale_factor)
                self.convolution_param['stride'] = int(scale_factor)
                self.convolution_param['group'] = self.inputs_shape[0][1]
                self.attrs = self.convolution_param
                # TODO: self.convolution_param['pads']
                self.weight = np.ones((self.outputs_shape[0][1], 1, int(scale_factor), int(scale_factor)), dtype=int)
                self.inputs_buf[1] = self.weight
                self.inputs_shape[1] = self.inputs_buf[1].shape
            else:
                self.upsample_param = dict()
                self.upsample_param['scale'] = scale_factor
                self.attrs = self.upsample_param
        elif self.mode == 'linear':
            self.interp_param = dict()
            self.interp_param['align_corners'] = True if coordinate == 'align_corners' else False
            self.attrs = self.interp_param

        self.setParsed()


    def convert(self):
        if self.mode == 'nearest':
            if hasattr(self, 'convolution_param'):
                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, None, convolution_param=self.convolution_param)
            elif hasattr(self, 'upsample_param'):
                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, upsample_param=self.upsample_param)
            else:
                raise NotImplementedError
        elif self.mode == 'linear':
                layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, interp_param=self.interp_param)
        else:
            raise NotImplementedError

        self.setConverted()

        return [layer]
