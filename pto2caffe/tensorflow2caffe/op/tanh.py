from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Tanh(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Tanh')
        self.setInited()


    def parse(self):
        self.layer_type = 'TanH'
        super().__parse__()

        # Attributes
        self.tanh_param = dict()
        self.attrs = self.tanh_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, tanh_param=self.tanh_param)

        self.setConverted()

        return [layer]
