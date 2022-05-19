from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class Silu(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code == 'F.silu')
        self.setInited()


    def parse(self):
        self.layer_type = 'Swish'
        super().__parse__()

        # Attributes
        self.swish_param = dict()
        self.swish_param['beta'] = 1.0

        self.attrs = self.swish_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, swish_param=self.swish_param)

        self.setConverted()

        return [layer]
