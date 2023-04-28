from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Flatten(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'flatten')
        self.setInited()


    def parse(self):
        self.type = 'Flatten'
        super().__parse__()

        self.flatten_param = dict()
        self.flatten_param['axis'] = self.inputs_buf[1]
        self.attrs = self.flatten_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, flatten_param=self.flatten_param)

        self.setConverted()

        return [layer]
