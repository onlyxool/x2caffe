from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class ReLUX(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator_code == 'RELU6')
        self.setInited()


    def parse(self):
        self.type = 'ReLUX'
        super().__parse__()

        # Attributes
        self.relux_param = dict()
        self.relux_param['negative_slope'] = 0
        self.relux_param['x'] = 6

        self.attrs = self.relux_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, relux_param=self.relux_param)

        self.setConverted()

        return [layer]
