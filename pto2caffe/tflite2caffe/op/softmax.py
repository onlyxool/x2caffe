import tflite

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class Softmax(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator == 'SOFTMAX')
        assert(self.op.InputsLength() == 1)
        assert(self.op.OutputsLength() == 1)
        self.setInited()


    def parse(self):
        self.layer_type = 'Softmax'

        self.parseInputOutput()

        self.softmax_param = dict()
        self.softmax_param['axis'] = 1

        self.attrs = self.softmax_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, softmax_param=self.softmax_param)

        self.setConverted()

        return [layer]
