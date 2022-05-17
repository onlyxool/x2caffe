import tflite

from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class Permute(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code == 'TRANSPOSE')
        assert(self.op.InputsLength() == 2)
        assert(self.op.OutputsLength() == 1)
        self.setInited()


    def parse(self):
        self.layer_type = 'Permute'

        self.parseInputOutput()

        self.permute_param = dict()
        self.permute_param['order'] = list(self.inputs_buf[1])

        self.attrs = self.permute_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, permute_param=self.permute_param)

        self.setConverted()

        return [layer]
