from caffe_transform import caffe_layer
from tvm2caffe.op.operator import Operator


class Sigmoid(Operator):

    def __init__(self, model, relay, index):
        super().__init__(model, relay, index)
        assert(self.operator_code == 'sigmoid')
        self.setInited()


    def parse(self):
        self.type = 'Sigmoid'
        super().__parse__()

        self.sigmoid_param = dict()
        self.attrs = self.sigmoid_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, sigmoid_param=self.sigmoid_param)

        self.setConverted()

        return [layer]
