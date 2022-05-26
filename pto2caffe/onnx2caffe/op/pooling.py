from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

from util import getLegacyAttrs
from onnx2caffe.utility import computePad


class Pooling(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code in ('MaxPool', 'AveragePool', 'GlobalAveragePool'))
        self.setInited()


    def parse(self):
        self.layer_type = 'Pooling'
        super().__parse__()

        # Pooling
        self.pooling_param = dict()
        if self.operator_code == 'MaxPool':
            self.pooling_param['pool'] = 0
            self.pooling_param['kernel_h'] = kernel_h = self.attrs['kernel_shape'][0]
            self.pooling_param['kernel_w'] = kernel_w = self.attrs['kernel_shape'][1]
        elif self.operator_code == 'AveragePool':
            self.pooling_param['pool'] = 1
            self.pooling_param['kernel_h'] = kernel_h = self.attrs['kernel_shape'][0]
            self.pooling_param['kernel_w'] = kernel_w = self.attrs['kernel_shape'][1]
        elif self.operator_code == 'GlobalAveragePool':
            self.pooling_param['pool'] = 1
            self.pooling_param['global_pooling'] = True
        else:
            raise NotImplementedError(self.operator_code)

        if 'dilations' in self.attrs and self.attrs['dilations'] != [1, 1]:
            raise NotImplementedError('Caffe Pooling don\'t support dilation')

        # Attributes
        kernel_size = self.attrs.get('kernel_shape', self.inputs_shape[0][-2:])
        strides = self.attrs.get('strides', [1, 1])
        self.pooling_param['stride_h'] = strides[0]
        self.pooling_param['stride_w'] = strides[1]
        self.pooling_param['ceil_mode'] = True if self.attrs.get('ceil_mode', False) else False

        # Padding
        legacy_pad = getLegacyAttrs(self, 'Pad')
        padding = computePad(self.type, self.attrs, self.inputs_shape[0], self.outputs_shape[0], kernel_size, strides, legacy_pad)
        self.pooling_param.update(padding)

        self.attrs = self.pooling_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]
