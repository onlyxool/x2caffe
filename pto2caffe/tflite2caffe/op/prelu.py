import tflite

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class PReLU(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code == 'PRELU')
        self.setInited()


    def parse(self):
        self.layer_type = 'PReLU'

        self.parseInputOutput()

        self.slope = self.inputs_buf[1].transpose(2, 0, 1)

        # Attributes
        self.prelu_param = dict()
        self.prelu_param['channel_shared'] = True if self.slope.shape[0] == 1 else False

        self.attrs = self.prelu_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, self.slope, prelu_param=self.prelu_param)

        self.setConverted()

        return [layer]
