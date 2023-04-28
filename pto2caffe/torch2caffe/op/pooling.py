from math import ceil

from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Pooling(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code in ('max_pool2d', 'avg_pool2d'))
        self.setInited()


    def compute_output_shape(self):
        if not self.isInputShapeFullyDefined(0):
            self.unSupported('Illegal Input Shape.')
            return

        n = self.inputs_shape[0][0]
        c = self.inputs_shape[0][1]
        if self.pooling_param['ceil_mode']:
            h = ceil((self.inputs_shape[0][2] + self.attrs['pad_h'] * 2 - self.attrs['kernel_h']) / self.attrs['stride_h']) + 1
            w = ceil((self.inputs_shape[0][3] + self.attrs['pad_w'] * 2 - self.attrs['kernel_w']) / self.attrs['stride_w']) + 1
        else:
            h = int((self.inputs_shape[0][2] + self.attrs['pad_h'] * 2 - self.attrs['kernel_h']) / self.attrs['stride_h']) + 1
            w = int((self.inputs_shape[0][3] + self.attrs['pad_w'] * 2 - self.attrs['kernel_w']) / self.attrs['stride_w']) + 1
 
        self.outputs_shape[0] = [n,c,h,w]
        self.model.tensor_shape[self.outputs[0]] = self.outputs_shape[0]


    def parse(self):
        self.type = 'Pooling'
        super().__parse__()

        self.pooling_param = dict()
        self.pooling_param['pool'] = 0 if self.operator_code == 'max_pool2d' else 1
        self.pooling_param['kernel_h'] = self.inputs_buf[1][0]
        self.pooling_param['kernel_w'] = self.inputs_buf[1][1]

        self.pooling_param['stride_h'] = self.inputs_buf[2][0]
        self.pooling_param['stride_w'] = self.inputs_buf[2][1]

        self.pooling_param['pad_h'] = self.inputs_buf[3][0]
        self.pooling_param['pad_w'] = self.inputs_buf[3][1]
        self.pooling_param['ceil_mode'] = self.inputs_buf[5]

        self.attrs = self.pooling_param

        self.compute_output_shape()

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, [self.inputs[0]], self.inputs_buf, self.outputs, self.weight, self.bias, pooling_param=self.pooling_param)

        self.setConverted()

        return [layer]
