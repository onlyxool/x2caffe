from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Softmax(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator == 'Softmax')
        self.setInited()


    def parse(self):
        self.layer_type = 'Softmax'
        super().__parse__()

        self.softmax_param = dict()
        self.attrs = self.softmax_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, softmax_param=self.softmax_param)

        self.setConverted()

        return [layer]
