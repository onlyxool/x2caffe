from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Flatten(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Flatten')
        self.setInited()


    def parse(self):
        self.layer_type = 'Flatten'
        super().__parse__()

        # Attributes
        self.flatten_param = dict()
        self.flatten_param['axis'] = self.attrs.get('axis', 1)

        self.attrs = self.flatten_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, flatten_param=self.flatten_param)
        self.setConverted()
        return [layer]

