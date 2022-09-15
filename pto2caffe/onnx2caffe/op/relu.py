from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class ReLU(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code in ('Relu', 'LeakyRelu'))
        self.setInited()


    def parse(self):
        self.type = 'ReLU'
        super().__parse__()

        # Attributes
        self.relu_param = dict()
        self.relu_param['negative_slope'] = self.attrs.get('alpha', 0)
        self.attrs = self.relu_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)

        self.setConverted()

        return [layer]
