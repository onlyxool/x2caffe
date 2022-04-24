from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class Sigmoid(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        self.sigmoid_param = dict()
        self.setInited()


    def parse(self):
        self.layer_type = 'Sigmoid'
        super().__parse__()

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, sigmoid_param=self.sigmoid_param)

        self.setConverted()

        return [layer]
