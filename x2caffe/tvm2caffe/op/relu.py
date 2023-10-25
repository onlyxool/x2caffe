from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class ReLU(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code in ('nn.relu', 'nn.leaky_relu'))
        self.setInited()


    def parse(self):
        super().__parse__()

        self.type = 'ReLU'
        self.relu_param = dict()
        self.relu_param['negative_slope'] = self.attrs.get('alpha', 0)
        self.attrs = self.relu_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)

        self.setConverted()

        return [layer]
