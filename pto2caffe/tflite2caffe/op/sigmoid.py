import tflite

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class Sigmoid(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code == 'LOGISTIC')
        self.setInited()


    def parse(self):
        self.type = 'Sigmoid'

        self.parseInputOutput()

        # Attributes
        self.sigmoid_param = dict()
        self.attrs = self.sigmoid_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, sigmoid_param=self.sigmoid_param)

        self.setConverted()

        return [layer]
