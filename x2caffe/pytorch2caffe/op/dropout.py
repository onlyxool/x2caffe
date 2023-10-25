from caffe_transform import caffe_layer
from pytorch2caffe.op.operator import Operator


class Dropout(Operator):

    def __init__(self, model, pnnx, type_code, index):
        super().__init__(model, pnnx, type_code, index)
        assert(self.operator_code == 'nn.Dropout')
        self.setInited()


    def parse(self):
        self.type = 'Dropout'
        super().__parse__()

        # Attributes
        self.dropout_param = dict()
        self.dropout_param['dropout_ratio'] = 0.5

        self.attrs = self.dropout_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, dropout_param=self.dropout_param)

        self.setConverted()

        return [layer]
