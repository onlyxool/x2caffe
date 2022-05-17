from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Softmax(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Softmax')
        self.setInited()


    def parse(self):
        self.layer_type = 'Softmax'
        super().__parse__()

        # Attributes
        self.softmax_param = dict()
        self.softmax_param['axis'] = self.attrs.get('axis', 1)
        self.attrs = self.softmax_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, softmax_param=self.softmax_param)

        self.setConverted()

        return [layer]
