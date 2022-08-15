from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator

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
            self.pooling_param['kernel_h'] = self.attrs['kernel_shape'][0]
            self.pooling_param['kernel_w'] = self.attrs['kernel_shape'][1]
        elif self.operator_code == 'AveragePool':
            self.pooling_param['pool'] = 1
            self.pooling_param['kernel_h'] = self.attrs['kernel_shape'][0]
            self.pooling_param['kernel_w'] = self.attrs['kernel_shape'][1]
        elif self.operator_code == 'GlobalAveragePool':
            self.pooling_param['pool'] = 1
            self.pooling_param['global_pooling'] = True

        if 'dilations' in self.attrs and self.attrs['dilations'] != [1, 1]:
            self.model.unsupport.append(self.operator_code)
            self.model.errorMsg.append('[' + self.node.name + ']: Operator ' + self.operator_code + ': Caffe Pooling don\'t support dilation.')
            return

        # Attributes
        kernel_size = self.attrs.get('kernel_shape', self.inputs_shape[0][-2:])
        self.pooling_param['stride_h'] = self.attrs.get('strides', [1, 1])[0]
        self.pooling_param['stride_w'] = self.attrs.get('strides', [1, 1])[1]
        self.pooling_param['ceil_mode'] = True if self.attrs.get('ceil_mode', False) else False

        # Padding
        legacy_pad = self.model.pad.get(self.inputs[0], {'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
        padding = computePad(self.type, self.attrs, self.inputs_shape[0], self.outputs_shape[0], kernel_size, self.attrs.get('strides', [1, 1]), legacy_pad)
        self.pooling_param.update(padding)

        self.attrs = self.pooling_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]
