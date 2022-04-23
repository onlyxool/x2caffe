from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class TanH(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.tanh_param = dict()
        self.setInited()


    def parse(self):
        self.layer_type = 'TanH'
        super().__parse__()

        # Attributes
        self.attrs = self.tanh_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, tanh_param=self.tanh_param)

        self.setConverted()

        return [layer]
