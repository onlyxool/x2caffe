from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Softplus(Operator):
    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Softplus')
        self.setInited()


    def parse(self):
        self.type = 'Softplus'
        super().__parse__()

        # Attributes
        self.softplus_param = dict()
        self.attrs = self.softplus_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs)

        self.setConverted()

        return [layer]
