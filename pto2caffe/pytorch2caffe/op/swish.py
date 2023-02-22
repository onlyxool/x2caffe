from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class Swish(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code == 'nn.Hardswish')
        self.setInited()


    def parse(self):
        self.type = 'Swish'
        super().__parse__()

        # Attributes
        self.swish_param = dict()
        if self.operator_code == 'nn.Hardswish':
            self.swish_param['beta'] = 1.0

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, swish_param=self.swish_param)

        self.setConverted()

        return [layer]
