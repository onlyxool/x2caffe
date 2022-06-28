from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Abs(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Abs')
        self.setInited()


    def parse(self):
        self.layer_type = 'AbsVal'
        super().__parse__()

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs)

        self.setConverted()

        return [layer]
