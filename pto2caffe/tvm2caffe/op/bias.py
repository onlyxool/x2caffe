from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator

from util import dim_map_nhwc2nchw


class Bias(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'nn.bias_add')
        self.setInited()


    def parse(self):
        super().__parse__()

        self.type = 'Bias'

        self.bias = self.inputs_buf[1]

        self.bias_param = dict()
        if 'axis' in self.attrs:
            self.bias_param['axis'] = self.attrs['axis']
        elif self.inputs_shape[0] is not None:
            self.bias_param['axis'] = self.inputs_shape[0].index(self.inputs_shape[1][0]) if len(self.inputs_shape[1]) > 0 else 0
        self.bias_param['axis'] = dim_map_nhwc2nchw[self.bias_param['axis']] if self.layout == 'NHWC' and len(self.inputs_shape[0]) == 4 else self.bias_param['axis']
        self.bias_param['num_axes'] = len(self.inputs_shape[1])

        self.attrs = self.bias_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.bias, bias_param=self.bias_param)

        self.setConverted()

        return [layer]
