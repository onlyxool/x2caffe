from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class LeakyRelu(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'LeakyRelu')
        self.setInited()


    def parse(self):
        self.layer_type = 'ReLU'
        super().__parse__()

        # Attribute
        self.relu_param = dict()
        self.relu_param['negative_slope'] = self.attrs.get('alpha', 0.3)
        self.attrs = self.relu_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.type, self.name, self.inputs, self.inputs_buf, self.outputs, relu_param=self.relu_param)

        self.setConverted()

        return [layer]
