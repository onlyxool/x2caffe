from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Debug(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        self.setInited()


    def parse(self):
        self.layer_type = self.op.type
        super().__parse__()

        self.debug()

        print(self.inputs_buf)
        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, debug_param=self.debug_param)

        self.setConverted()

        return [layer]
