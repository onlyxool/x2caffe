from caffe_transform import caffe_layer
from onnx2caffe.op.operator import Operator


class Exp(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, node, index)
        assert(self.operator_code == 'Exp')
        self.setInited()


    def parse(self):
        self.layer_type = 'Exp'
        super().__parse__()

        # Attributes
        # Leave all arguments in ExpParameter as default
        # base = -1.0 (base = e)
        # scale = 1.0
        # shift = 0.0

        self.exp_param = dict()
        self.attrs = self.exp_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, exp_param=self.exp_param)

        self.setConverted()

        return [layer]
