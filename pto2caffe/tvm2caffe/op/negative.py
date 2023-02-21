from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Negative(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'negative')
        self.setInited()


    def parse(self):
        super().__parse__()

        self.type = 'Scale'

        self.scale_param = dict()
        self.scale_param['bias_term'] = False
        self.scale_param['axis'] = 0
        self.scale_param['num_axes'] = 0

        self.weight = -1
        self.bias = None

        self.attrs = self.scale_param
        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, self.weight, self.bias, scale_param=self.scale_param)

        self.setConverted()

        return [layer]
