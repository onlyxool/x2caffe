from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator
from util import handleLegacyPad


class MaxPoolWithArgmax(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'MaxPoolWithArgmax')
        self.setInited()


    def parse(self):
        self.layer_type = 'Pooling'
        super().__parse__()

        # Attribute
        self.pooling_param = dict()
        self.pooling_param['pool'] = 0
        self.pooling_param['kernel_h'] = self.attrs['ksize'][self.ndim('H')]
        self.pooling_param['kernel_w'] = self.attrs['ksize'][self.ndim('W')]
        self.pooling_param['stride_h'] = self.attrs['strides'][self.ndim('H')]
        self.pooling_param['stride_w'] = self.attrs['strides'][self.ndim('W')]
        self.pooling_param['ceil_mode'] = True if self.attrs['padding'] == 'SAME' else False

        # Padding
        legacy_pad = self.model.pad.get(self.inputs[0], {'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
        padding = handleLegacyPad(self.attrs['padding'], self.inputs_shape[0], self.outputs_shape[0], self.pooling_param, legacy_pad, self.type)
        self.pooling_param.update(padding)

        self.attrs = self.pooling_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]
