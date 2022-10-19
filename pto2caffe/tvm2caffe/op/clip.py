from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Clip(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'clip')
        self.setInited()


    def parse(self):
        super().__parse__()

        self.type = 'ReLUX'
        self.relux_param = dict()
        self.relux_param['negative_slope'] = self.attrs.get('a_min', 0)
        self.relux_param['x'] = self.attrs.get('a_max', 0)
        self.attrs = self.relux_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, relux_param=self.relux_param)

        self.setConverted()

        return [layer]
