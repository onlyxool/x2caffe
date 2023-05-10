from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Exp(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'exp')
        self.setInited()


    def parse(self):
        self.type = 'Exp'
        super().__parse__()

        self.exp_param = dict()
        self.attrs = self.exp_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, exp_param=self.exp_param)

        self.setConverted()

        return [layer]
