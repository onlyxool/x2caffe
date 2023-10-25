from caffe_transform import caffe_layer
from tensorflow2caffe.op.operator import Operator


class Elu(Operator):

    def __init__(self, model, tf_op, index):
        super().__init__(model, tf_op, index)
        assert(self.operator_code == 'Elu')
        self.setInited()


    def parse(self):
        self.type = 'Elu'
        super().__parse__()

        self.elu_param = dict()
        self.elu_param['alpha'] = 1.0

        self.attrs = self.elu_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, elu_param=self.elu_param)

        self.setConverted()

        return [layer]
