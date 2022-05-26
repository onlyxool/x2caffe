from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

from util import getLegacyAttrs
from onnx2caffe.utility import computePad


class Deconvolution(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'ConvTranspose')
        self.setInited()


    def parse(self):
        self.layer_type = 'Deconvolution'
        super().__parse__()

        # Weight
        self.weight = self.inputs_buf[1]

        # Bias
        self.bias = self.inputs_buf[2] if len(self.inputs_buf) == 3 else None

        # Attributes
        self.convolution_param = dict()
        self.convolution_param['num_output'] = self.outputs_shape[0][1]

        strides = self.attrs.get('strides', [1, 1])
        self.convolution_param['stride_h'] = strides[0]
        self.convolution_param['stride_w'] = strides[1]

        self.convolution_param['dilation'] = self.attrs.get('dilations', [1, 1])
        self.convolution_param['group'] = self.attrs.get('group', 1)
        self.convolution_param['kernel_size'] = kernel_size = self.attrs['kernel_shape']
        self.convolution_param['bias_term'] = True if self.bias is not None else False

        # Padding
        legacy_pad = getLegacyAttrs(self, 'Pad')
        padding = computePad(self.type, self.attrs, self.inputs_shape[0], self.outputs_shape[0], kernel_size, strides, legacy_pad)
        self.convolution_param.update(padding)

        self.attrs = self.convolution_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, convolution_param=self.convolution_param)

        self.setConverted()

        return [layer]
