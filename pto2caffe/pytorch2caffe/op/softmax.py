from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class Softmax(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator == 'nn.Softmax')
        self.setInited()


    def parse(self):
        self.layer_type = 'Softmax'
        super().__parse__()

        # Attributes
        self.softmax_param = dict()

        self.softmax_param['axis'] = self.attrs['dim']
        self.attrs = self.softmax_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, softmax_param=self.softmax_param)

        self.setConverted()

        return [layer]
