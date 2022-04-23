from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Concat(Operator):
    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        self.setInited()


    def parse(self):
        self.layer_type = 'Concat'
        super().__parse__()

        # Attributes
        self.concat_param = dict()
        self.concat_param['axis'] = self.attrs['axis']

        self.attrs = self.concat_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, concat_param=self.concat_param)

        self.setConverted()

        return [layer]
