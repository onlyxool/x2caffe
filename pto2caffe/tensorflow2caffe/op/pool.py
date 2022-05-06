from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator
from util import handleLegacyPad


class Pool(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator == 'MaxPool' or self.operator == 'AvgPool')
        self.setInited()


    def parse(self):
        self.layer_type = 'Pooling'
        super().__parse__()

        # Attribute
        self.pooling_param = dict()
        if self.operator == 'MaxPool':
            self.pooling_param['pool'] = 0
        elif self.operator == 'AvgPool':
            self.pooling_param['pool'] = 1
        else:
            raise NotImplementedError(self.operator)

        self.pooling_param['kernel_h'] = self.attrs['ksize'][self.ndim('H')]
        self.pooling_param['kernel_w'] = self.attrs['ksize'][self.ndim('W')]
        self.pooling_param['stride_h'] = self.attrs['strides'][self.ndim('H')]
        self.pooling_param['stride_w'] = self.attrs['strides'][self.ndim('W')]
        self.pooling_param['ceil_mode'] = True if self.attrs['padding'] == 'SAME' else False

        # Padding
        legacy_pad = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
        for legacy in self.model.legacys:
            if legacy.operator == 'Pad':
                if legacy.outputs[0] == self.inputs[0]:
                    legacy_pad = legacy.pad
                    self.inputs[0] = legacy.inputs[0]
                    self.inputs_shape[0] = legacy.inputs_shape[0]

        padding = handleLegacyPad(self.attrs['padding'], self.inputs_shape[0], self.outputs_shape[0], self.pooling_param, legacy_pad, self.type)
        if len(padding) == 2:
            self.pooling_param['pad_w'] = padding[0]
            self.pooling_param['pad_h'] = padding[1]
        elif len(padding) == 4:
            self.pooling_param['pad_l'] = padding[0]
            self.pooling_param['pad_r'] = padding[1]
            self.pooling_param['pad_t'] = padding[2]
            self.pooling_param['pad_b'] = padding[3]

        self.attrs = self.pooling_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]
