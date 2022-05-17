from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Reshape(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code in ('Reshape', 'Squeeze'))
        self.setInited()


    def parse(self):
        self.layer_type = 'Reshape'
        super().__parse__()

        self.reshape_param = dict(shape=dict(dim=self.outputs_shape[0]))
        self.attrs = self.reshape_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, reshape_param=self.reshape_param)

        self.setConverted()

        return [layer]
