from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class ReLU(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code == 'RELU')
        self.setInited()


    def parse(self):
        self.type = 'ReLU'

        if self.op is not None:
            self.parseInputOutput()

        # Attributes
        self.relu_param = dict()
        self.relu_param['negative_slope'] = 0
        self.attrs = self.relu_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)

        self.setConverted()

        return [layer]
