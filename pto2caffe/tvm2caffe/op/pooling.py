import numpy as np

from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Pooling(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code in ('nn.max_pool2d', 'nn.avg_pool2d', 'nn.global_avg_pool2d'))
        self.setInited()


    def parse(self):
        self.type = 'Pooling'
        super().__parse__()

        if 'dilation' in self.attrs and self.attrs['dilation'] != (1, 1):
            self.unSupported('Caffe Pooling don\'t support dilation.')
            return

        # Attributes
        self.pooling_param = dict()
        self.pooling_param['pool'] = 0 if self.operator_code == 'nn.max_pool2d' else 1
        if self.operator_code == 'nn.global_avg_pool2d':
            self.pooling_param['stride_h'] = 1
            self.pooling_param['stride_w'] = 1
            self.pooling_param['ceil_mode'] = False
            self.pooling_param['global_pooling'] = True
        else:
            self.pooling_param['kernel_h'] = self.attrs['pool_size'][0]
            self.pooling_param['kernel_w'] = self.attrs['pool_size'][1]
            self.pooling_param['stride_h'] = self.attrs.get('strides', [1, 1])[0]
            self.pooling_param['stride_w'] = self.attrs.get('strides', [1, 1])[1]
            self.pooling_param['ceil_mode'] = self.attrs.get('ceil_mode', False)

        # Padding
        legacy_pad = self.model.pad.get(self.relay_inputs[0], [0, 0, 0, 0])
        attr_pad = self.attrs.get('padding', [0, 0, 0, 0])
        pool_pad = (np.array(legacy_pad) + np.array(attr_pad)).tolist()
        if pool_pad[0] == pool_pad[2] and pool_pad[1] == pool_pad[3]:
            self.pooling_param['pad_h'] = pool_pad[0]
            self.pooling_param['pad_w'] = pool_pad[1]
        else:
            self.pooling_param['pad_t'] = pool_pad[0]
            self.pooling_param['pad_l'] = pool_pad[1]
            self.pooling_param['pad_b'] = pool_pad[2]
            self.pooling_param['pad_r'] = pool_pad[3]

        self.attrs = self.pooling_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]
