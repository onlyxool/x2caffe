from caffe_transform import caffe_layer
from torch2caffe.op.operator import Operator


class Dropout(Operator):

    def __init__(self, model, node, index):
        super().__init__(model, model.graph, node, index)
        assert(self.operator_code == 'dropout')
        self.setInited()


    def parse(self):
        self.type = 'Dropout'
        super().__parse__()

        self.dropout_param = dict()
        self.dropout_param['dropout_ratio'] = self.inputs_buf[1]
        self.attrs = self.dropout_param

        self.setParsed()


    def convert(self):
        layer = caffe_layer(self.layer_type, self.name, self.inputs, self.inputs_buf, self.outputs, dropout_param=self.dropout_param)

        self.setConverted()

        return [layer]
