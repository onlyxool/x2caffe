from caffe_transform import caffe_layer
from tflite2caffe.op.operator import Operator


class Reshape(Operator):

    def __init__(self, model, tf_op, tf_op_name, index):
        super().__init__(model, tf_op, tf_op_name, index)
        assert(self.operator in ('RESHAPE', 'SQUEEZE'))
        self.setInited()


    def parse(self):
        self.layer_type = 'Reshape'

        self.parseInputOutput()

        # Attributes
        self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))

        self.attrs = self.reshape_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)

        self.setConverted()

        return [layer]
