from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Tanh(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'tanh')
        self.setInited()


    def parse(self):
        self.type = 'TanH'
        super().__parse__()

        # Attributes
        self.tanh_param = dict()
        self.attrs = self.tanh_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, tanh_param=self.tanh_param)

        self.setConverted()

        return [layer]
