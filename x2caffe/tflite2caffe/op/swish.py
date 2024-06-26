from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class Swish(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code == 'HARD_SWISH')
        assert(self.op.InputsLength() == 1)
        assert(self.op.OutputsLength() == 1)

        self.setInited()


    def parse(self):
        self.type = 'HardSwish'
        super().__parse__()

        self.swish_param = dict()
        self.attrs = self.swish_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, swish_param=self.swish_param)

        self.setConverted()

        return [layer]
