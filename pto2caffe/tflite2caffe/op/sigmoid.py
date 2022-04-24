import tflite

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class Sigmoid(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        self.setInited()


    def parse(self):
        self.layer_type = 'Sigmoid'
        assert(self.operator == 'LOGISTIC')

        self.parseInput()
        self.parseOutput()

        # Attributes
        self.sigmoid_param = dict()
        self.attrs = self.sigmoid_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, sigmoid_param=self.sigmoid_param)

        self.setConverted()

        return [layer]
